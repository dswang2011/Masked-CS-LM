
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(opt.bert_path)

texts = ["test some text for language model.", "another test example for it"]

inputs = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

inputs['labels'] = inputs.input_ids.detach.clone()


# Now we mask tokens in the input_ids tensor, using the 15% probability we used before - 
# and the not a CLS (101) or SEP (102) token condition, and PAD tokens (0 input ids).

# create random array of floats with equal dimensions to input_ids tensor


def masked_inputs(batch_encodings):
    rand_mat = torch.rand(batch_encodings.input_ids.shape)
    # create mask array
    threshold = (rand_mat < 0.50)
    cnt = 0
    while torch.all(threshold[:,:192]):
        rand_mat = torch.rand(batch_encodings.input_ids.shape)
        threshold = (rand_mat < 0.50)
        cnt+=1
        if cnt>10: break
        
    mask_arr = threshold * (inputs.input_ids != 101) * \
            (batch_encodings.input_ids != 102) * (inputs.input_ids != 0)


    # now we take the indices of each True value, for each vector.
    selection = []  # positions that are masked
    for i in range(batch_encodings.input_ids.shape[0]):
        select = torch.flatten(mask_arr[i].nonzero()).tolist()
        select = [item for item in select if item < 192]
        selection.append(selection)


    # Step2: Apply these indices to each respective row in input_ids, assigning [MASK] positions as 103.
    for i in range(batch_encodings.input_ids.shape[0]):
        batch_encodings.input_ids[i, selection[i]] = 103

    return selection, inputs



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


dataset = MyDataset(inputs)


loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)




from tqdm import tqdm  # for our progress bar

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

