import torch
from torch import nn
import transformers 
from transformers import DistilBertTokenizer, DistilBertPreTrainedModel, DistilBertModel
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import datasets
import numpy as np

from tqdm import tqdm

MASK_ID = 103




def experiment_data_builder(text, treatment, confounder, Y = None, tokenizer = None):
    ### Assemble a dataset
    ### Given text vector, labels treatment and confounder
    ### Performing Masking and tokenizing 
    if Y is None:
        Y = [-1 for _ in range(len(treatment))]

    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased', do_lower_case=True)

    ### Create a dictionary
    output_data = {"text": text,
                   "treatment": treatment,
                   "confounder": confounder,
                   "Y":Y}
    if output_data["Y"] is None:
        output_data["Y"] = [-1] * len(output_data["treatment"])
    ### tokenize the text
    output_data = datasets.Dataset.from_dict(output_data)
    output_data = output_data.map(encode_data, fn_kwargs = {"tokenizer": tokenizer}, batched = True)
    output_data = output_data.map(get_length,  batched = False)

    ### split into test val train split
    train_val_split = output_data.train_test_split(test_size = 0.25)
    val_test_split = train_val_split["test"].train_test_split(test_size = 0.67)
    
    output_data = datasets.DatasetDict({"train":train_val_split["train"],
                                        "val":val_test_split["train"],
                                        "test":val_test_split["test"]})
    

    output_data = output_data.with_format('torch')

    return(output_data)

def encode_data(observation, tokenizer, column = "text"):
    output = tokenizer.batch_encode_plus(observation[column], add_special_tokens = True,\
                                   max_length = 128,
                                   pad_to_max_length = True)

    return(output)

def get_length(observation):
    input_id_lists = observation["input_ids"]
    candidate_pads = np.where(np.array(input_id_lists)==0)[0]
    if len(candidate_pads) > 0:
        first_pad_index = candidate_pads[0]
    else:
        first_pad_index = 128
    return({"length": first_pad_index})



class cause_bert_model(DistilBertPreTrainedModel):
    def __init__(self,config):
        ## pass in the configuration of the model
        super().__init__(config)

        if torch.has_cuda:
            self.mydevice = "cuda"
        elif torch.has_mps:
            self.mydevice = "mps"
        else:
            self.mydevice = None

        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size

        ### set up the bert adjustment model
        self.bert_adjustment_model = DistilBertModel(config)

        ### This is the projection of mlm part
        self.projection_to_vocab = nn.Sequential(nn.Linear(self.config.dim, self.config.dim),
                                                 nn.GELU(),
                                                 nn.LayerNorm(self.config.dim, eps = 1e-12),
                                                 nn.Linear(self.config.dim, self.vocab_size))
        def weights_init(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


        ## initialize the weights for the projection layer
        self.projection_to_vocab.apply(weights_init)
        
        self.Q_cls = nn.ModuleDict()
        
        for label in ["0", "1"]:
            self.Q_cls[label] = nn.Sequential(
                nn.Linear(config.hidden_size + 2, 200), ### for replicate purpose we set to be 2
                nn.ReLU(),
                nn.Linear(200, 2)) ### for replicate purpose we set to be 2
            
        self.Q_cls["0"].apply(weights_init)
        self.Q_cls["1"].apply(weights_init)
        ## Initialize weights for the model
        self.init_weights()
        self.cross_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim =1)


    def forward(self, input_ids, length_sentence, mask, confound,\
                 treatment, outcome = None, mlm_val = True):
        ## input_ids: B * d
        ## length_sentence: B
        ## mask: B * d 
        ## confound: B * 2 (requires one-hot encoding)
        ## treatment: B
        ## outcome: B
        batch_size = input_ids.shape[0]

        if self.mydevice is not None:
            input_ids = input_ids.to(self.mydevice)
            length_sentence = length_sentence.to(self.mydevice)
            mask = mask.to(self.mydevice)
            confound = confound.to(self.mydevice)
            treatment = treatment.to(self.mydevice)
            outcome = outcome.to(self.mydevice)
        
        if mlm_val:
            ### produce masks for masked language modeling
            ### The authors only mask one token at random
            length_sentence = length_sentence.unsqueeze(1) - 2 
            
            ## The above covers 0 to the last position-1 that has a value
            ## We should plus 1 to make sure that CLS is not masked and the last token code be masked
            ## Generate index to mask for each token

            mask_positions = torch.round(torch.rand(size = (batch_size,1)).to(self.mydevice) *
                                         (length_sentence)).long() + 1

            ## get the labels being masked
            correct_labels = torch.gather(input_ids,1, mask_positions)
            mlm_eval_ground_truth = (torch.ones(size = input_ids.shape) * (-100)).long()

            if self.mydevice is not None:
                mlm_eval_ground_truth = mlm_eval_ground_truth.to(self.mydevice)

            ### Assign back the ground truth for mlm evaluation
            mlm_eval_ground_truth.scatter_(1, mask_positions, correct_labels)
            ### mask the label in inputs
            
            input_ids.scatter_(1, mask_positions, MASK_ID)

        ## pass the input through bert
        full_output = self.bert_adjustment_model(input_ids, attention_mask = mask)[0]
        cls_output = full_output[:,0,:]

        ## onehot encode the confounders
        confound = nn.functional.one_hot(confound, 2)

        ### Evaluate the mlm loss
        if mlm_val:
            prediction = self.projection_to_vocab(full_output) ## output is * vocab size 
            mlm_loss = self.cross_loss(prediction.view(-1, self.vocab_size),
                                           mlm_eval_ground_truth.view(-1))
        else:
            mlm_loss = 0.0
            
        Y_classifier_input = torch.cat((cls_output,confound), dim = 1)

        ### Evaluate the Classifier Loss
        Y_classfier_T_1 = self.Q_cls['1'](Y_classifier_input)
        Y_classfier_T_0 = self.Q_cls['0'](Y_classifier_input)


        ### Evaluate loss
        if outcome is None or torch.all(outcome == -1):
            ### If no given outcome or outcome doesn't exist
            Y_loss = 0.0
        else:
            ## get the correct mask for label
            T0_indices = torch.where(treatment == 0)[0]
            
            T1_indices = torch.where(treatment == 1)[0]

            ## let model evaluate only the corresponding term while evaluation by replacing irrelevant terms to -100
            if T0_indices.size(dim = 0) == 0:
                outcome_T1_eval = outcome.clone()
            else:
                outcome_T1_eval = outcome.clone().scatter(0,T0_indices, -100)

            if T1_indices.size(dim = 0) == 0:
                outcome_T0_eval = outcome.clone()
            else:
                outcome_T0_eval = outcome.clone().scatter(0,T1_indices, -100)

            ## the loss is the sum of two terms
            Y_loss = self.cross_loss(Y_classfier_T_1.view(-1, 2), outcome_T1_eval) + \
                self.cross_loss(Y_classfier_T_0.view(-1, 2), outcome_T0_eval)

        ## Get the hypothetical estimate Q1 and Q0
        Q0 = self.softmax(Y_classfier_T_0)[:, 1] ## prob for the Y=1 outcome given T = 0
        Q1 = self.softmax(Y_classfier_T_1)[:, 1] ## prob for the Y=1 outcome given T = 1
        return(Q0, Q1, Y_loss, mlm_loss)
    
    
class training_cause_text:
    def __init__(self, Y_weight = 0.1, mlm_weight = 1, batch_size = 32):
        ### Load the weight
        self.mlm_weight = mlm_weight
        self.Y_weight = Y_weight

        ### Load the model
        self.distilBERT_Cause = cause_bert_model.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False)
        
        self.device = None
        
        if torch.has_cuda:
            self.device = 'cuda'
            self.distilBERT_Cause.to(self.device)
        elif torch.has_mps:
            self.device = 'mps'
            self.distilBERT_Cause.to(self.device)

        self.batch_size = batch_size
    
    def train(self, dataset, valset, lr = 2e-5, epoches = 3, monitor_step = 30):
        ## create two data loaders: train and eval
        ## eval was done after every epoch
        val_loader = self.create_data_loader(valset, sampler = "sequential")
        train_loader = self.create_data_loader(dataset)

        ### Set up optimizer and schedulers as suggested by the paper and the code
        ### I refer to the original code on this since the data doesn't specify about the warm up and the schdeduler
        optimizer = AdamW(self.distilBERT_Cause.parameters(), lr=lr, eps=1e-8)
        total_steps = len(train_loader) * epoches
        warmup_steps = total_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        for epoch in range(epoches):
            self.distilBERT_Cause.train()
            for step, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
                ### Here are the orginal data
                T, C, Y, text_ids  = batch["treatment"], batch["confounder"], batch["Y"], batch["input_ids"]
                ### Here are the other auxillary inputs for classification
                sen_len, attn_masks  = batch["length"], batch["attention_mask"]

                self.distilBERT_Cause.zero_grad()
                ### do an iteration
                _, _, Y_loss, mlm_loss = self.distilBERT_Cause(text_ids, sen_len, attn_masks, C, T, Y, mlm_val = True)

                loss = self.Y_weight * Y_loss + self.mlm_weight * mlm_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                ### loss monitor
                if (step + 1) % monitor_step == 0:
                    print("At training step",step, "of epoch", epoch, \
                          "the loss is:", loss.detach().cpu().item())

            print("Start to evaluate epoch", epoch)   
            current_val_loss = self.eval(val_loader)
            print("The current loss for epoch", epoch, " is", current_val_loss.cpu().numpy())   


    def eval(self, val_loader):
        ### Add evaluation code in after each epoch
        loss = torch.tensor([0.0])
        batch_count = torch.tensor([0.0])
        if self.device is not None:
            loss = loss.to(self.device)
            batch_count = batch_count.to(self.device)
        self.distilBERT_Cause.eval()
        for step, batch in tqdm(enumerate(val_loader), total = len(val_loader)):
                ### Here are the orginal data
                T, C, Y, text_ids  = batch["treatment"], batch["confounder"], batch["Y"], batch["input_ids"]
                ### Here are the other auxillary inputs for classification
                sen_len, attn_masks  = batch["length"], batch["attention_mask"]

                ### do an iteration
                _, _, Y_loss, mlm_loss = self.distilBERT_Cause(text_ids, sen_len, attn_masks, C, T, Y, mlm_val = True)

                loss += (self.Y_weight * Y_loss.detach() + self.mlm_weight * mlm_loss.detach()) * len(batch) 
                batch_count += len(batch)
        return loss/batch_count
    

    def test(self, testset):
        ### assemble test_set to loader
        test_loader = self.create_data_loader(testset, sampler = "sequential")

        ### the records of the Qs
        Q0_records = []
        Q1_records = []
        self.distilBERT_Cause.eval()
        for step, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
                ### Here are the orginal data
                T, C, Y, text_ids  = batch["treatment"], batch["confounder"], batch["Y"], batch["input_ids"]
                ### Here are the other auxillary inputs for classification
                sen_len, attn_masks  = batch["length"], batch["attention_mask"]

                ### do an iteration
                Q0, Q1, _, _ = self.distilBERT_Cause(text_ids, sen_len, attn_masks, C, T, Y, mlm_val = False)

                Q0_records += Q0.detach().cpu().numpy().tolist()
                Q1_records += Q1.detach().cpu().numpy().tolist()
        
        return Q0_records, Q1_records
    
    def ATE_text_adjust(self, Q0_records, Q1_records):
        results_prob = np.array(list(zip(Q0_records, Q1_records)))
        ### Their source code is incorrect: they do Q0-Q1
        return(np.mean(results_prob[:,1]- results_prob[:,0]))
        
    def create_data_loader(self, dataset, sampler = "random"):
        ## create a data sampler
        ## This part largely mimics the original source code
        sampler = RandomSampler(dataset) if sampler == 'random' else SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return(data_loader)

        







        

            

            

        

        



    

    

