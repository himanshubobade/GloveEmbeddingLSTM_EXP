# Load GloVe embeddings
glove_path = 'glove.6B.100d.txt'
glove_vectors = {}
embedding_dim = 100

# Initialize special tokens
special_tokens = ['[PAD]', '[UNK]']

with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vector = [float(val) for val in parts[1:]]
        glove_vectors[word] = vector

# Insert special tokens at the beginning of the vocabulary
for token in special_tokens:
    glove_vectors[token] = [0.0] * embedding_dim  # Use zero vectors for special tokens

# Create word2idx mapping
word2idx = {word: idx for idx, word in enumerate(glove_vectors)}

# Initialize word embeddings with GloVe vectors
embedding_matrix = torch.FloatTensor([glove_vectors[word] for word in glove_vectors])

# Create embedding layer with GloVe embeddings
embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=word2idx['[PAD]'], freeze=True)

# Define casing feature extraction function
def extract_casing_features(word):
    features = [
        word[0].isupper(),          # Start with a capital letter
        word.isupper()              # All capital letters
        # word.islower(),             # All lower case letters
        # word.isdigit()              # All digits
        # any(c.isalpha() for c in word) and any(c.isdigit() for c in word)  # Mix of words and digits
    ]
    return torch.tensor(features, dtype=torch.float32)

# Lowercase tokens and add casing features
dataset = (
    dataset
    .map(lambda x: {
            'input_ids': [
                word2idx.get(word.lower(), word2idx['[UNK]'])
                for word in x['tokens']
            ],
            'casing_features': [
                extract_casing_features(word)
                for word in x['tokens']
            ]
        }
    )
)

# Define a modified collate function to pad sequences and labels with 9 for 0 in input_ids
def collate_fn(batch):
    max_seq_len = max(len(item['input_ids']) for item in batch)
    padded_input_ids = [item['input_ids'] + [0] * (max_seq_len - len(item['input_ids'])) for item in batch]
    padded_casing_features = [item['casing_features'] + [[0] * 2] * (max_seq_len - len(item['casing_features'])) for item in batch]
    padded_labels = [item['labels'] + [9] * (max_seq_len - len(item['labels'])) for item in batch]
    return {
        'input_ids': torch.tensor(padded_input_ids),
        'casing_features': torch.tensor(padded_casing_features),
        'labels': torch.tensor(padded_labels)
    }

# Create DataLoader for training, validation, and test datasets using the modified collate function
train_loader = DataLoader(dataset['train'], batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(dataset['validation'], batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(dataset['test'], batch_size=32, collate_fn=collate_fn)

# Define your configuration
class GloveEmbeddedBidirectionalLSTMConfig:
    word_vocab_size = len(word2idx)
    embedding_dim = 100
    lstm_hidden = 256
    lstm_dropout = 0.33
    linear_output_dim = 128
    num_classes = len(idx2tag)
    input_pad_token_id = 0
    label_pad_token_id = 9
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 22

# Instantiate the model
class GloveEmbeddedBidirectionalLSTM(nn.Module):
    def __init__(self, config):
        super(GloveEmbeddedBidirectionalLSTM, self).__init__()

        # Create embedding layer with GloVe embeddings
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=word2idx['[PAD]'], freeze=True)
        self.lstm = nn.LSTM(config.embedding_dim + 2, config.lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.lstm_dropout)
        self.linear = nn.Linear(2 * config.lstm_hidden, config.linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(config.linear_output_dim, config.num_classes)

    def forward(self, input_ids, casing_features):
        # Modify the forward pass to include casing features
        x = self.embedding(input_ids)

        # Concatenate casing features with the embedded tokens
        x = torch.cat((x, casing_features), dim=-1)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        logits = self.classifier(elu_out)
        return logits

# Instantiate the model
model = GloveEmbeddedBidirectionalLSTM(GloveEmbeddedBidirectionalLSTMConfig())
model = model.to('cuda')

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=GloveEmbeddedBidirectionalLSTMConfig.label_pad_token_id)

# Define optimizer with the learning rate
optimizer = optim.Adam(model.parameters(), lr=GloveEmbeddedBidirectionalLSTMConfig.learning_rate, weight_decay=1e-5)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop with overfitting checks
train_losses = []
valid_losses = []

# Training loop
for epoch in range(GloveEmbeddedBidirectionalLSTMConfig.num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch['input_ids'].to('cuda')
        casing_features = batch['casing_features'].to('cuda')
        labels = batch['labels'].to('cuda')

        optimizer.zero_grad()

        # Add casing_features to the model input
        outputs = model(input_ids, casing_features)

        loss = criterion(outputs.view(-1, GloveEmbeddedBidirectionalLSTMConfig.num_classes), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Learning rate scheduling step
    scheduler.step()

    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    # Validation loss
    model.eval()
    with torch.no_grad():
        total_valid_loss = 0.0
        for valid_batch in valid_loader:
            valid_input_ids = valid_batch['input_ids'].to('cuda')
            valid_casing_features = valid_batch['casing_features'].to('cuda')
            valid_labels = valid_batch['labels'].to('cuda')

            # Add casing_features to the model input
            valid_outputs = model(valid_input_ids, valid_casing_features)

            valid_loss = criterion(valid_outputs.view(-1, GloveEmbeddedBidirectionalLSTMConfig.num_classes), valid_labels.view(-1))
            total_valid_loss += valid_loss.item()

    average_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(average_valid_loss)
    print(f"Epoch [{epoch + 1}/{GloveEmbeddedBidirectionalLSTMConfig.num_epochs}] - "
          f"Training Loss: {average_loss:.4f}, Validation Loss: {average_valid_loss:.4f}")

    # Early stopping check (you can modify this based on your needs)
    if epoch > 5 and valid_losses[-1] > valid_losses[-2]:
        print("Early stopping as validation loss started increasing.\n")
        break

print()

# Plot the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


def evaluate_model_with_padding(model, dataloader, idx2tag, ignore_index=9):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to('cuda')
            casing_features = batch['casing_features'].to('cuda')
            labels = batch['labels'].to('cuda')

            # Add casing_features to the model input
            outputs = model(input_ids, casing_features)

            preds = torch.argmax(outputs, dim=2)

            # Remove predictions for padded tokens
            for i in range(len(input_ids)):
                non_padding_mask = (input_ids[i] != GloveEmbeddedBidirectionalLSTMConfig.input_pad_token_id).tolist()
                pred_seq = preds[i][non_padding_mask].tolist()
                label_seq = labels[i][non_padding_mask].tolist()

                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    # Map the labels and predictions to tag strings
    labels = [
        list(map(idx2tag.get, label_seq))
        for label_seq in all_labels
    ]
    preds = [
        list(map(idx2tag.get, pred_seq))
        for pred_seq in all_preds
    ]

    # Evaluate precision, recall, and F1 score
    precision, recall, f1 = evaluate(
        list(itertools.chain(*labels)),
        list(itertools.chain(*preds)),
    )

    return precision, recall, f1

print()