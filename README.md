# Text-to-SQL Transformer: Natural Language to SQL Query Generation Using a Transformer Built from Scratch

## 1. Project Overview

The **Text-to-SQL Transformer** is a deep learning project that converts natural language questions into structured SQL queries. The main objective is to enable users to interact with relational databases using everyday language instead of writing SQL manually.

For example, a user may ask:

> What is the nationality of Terrence Ross?

The system processes the question along with the table's column names and generates a SQL query:

```sql
SELECT Nationality
WHERE Player = 'terrence ross'
```

The project uses the WikiSQL dataset and implements a Transformer encoder-decoder architecture from scratch using PyTorch. The model learns to understand the relationship between natural language questions, database schemas, and SQL query structures.

Unlike approaches that use pretrained language models or ready-made Transformer implementations, this project builds the essential Transformer components manually, including scaled dot-product attention, multi-head attention, feed-forward networks, encoder layers, decoder layers, positional encoding, and attention masks.

The project also includes data preprocessing, tokenization, model training, greedy decoding, beam search, SQL conversion, official evaluation, visualizations, and a Streamlit-based user interface.

## 2. Problem Statement

Traditional database interaction requires users to understand SQL syntax and database structures. Users who do not have SQL knowledge may find it difficult to retrieve information from databases.

For example, to retrieve the nationality of a particular player, a user may need to write a query such as:

```sql
SELECT Nationality
FROM Player
WHERE Player = 'terrence ross';
```

The project aims to reduce this difficulty by allowing users to express their requirements in natural language.

However, translating natural language into SQL is challenging because the system must:

- Understand the user's question.
- Identify the relevant database column.
- Determine the appropriate SQL aggregation operation.
- Identify filtering conditions.
- Select the correct comparison operator.
- Handle multiple conditions.
- Generate SQL in the correct order and format.
- Interpret the database schema provided with the question.

The proposed system addresses these challenges using a Transformer-based sequence-to-sequence model.

## 3. Project Objectives

The main objectives of this project are:

1. To develop a Text-to-SQL system using the WikiSQL dataset.
2. To implement a Transformer architecture from scratch using PyTorch.
3. To understand and implement the core components of the Transformer.
4. To preprocess natural language questions and SQL queries into a suitable training format.
5. To train a model that maps natural language questions and table schemas into SQL representations.
6. To implement greedy decoding and beam search with a beam size of four.
7. To convert generated SQL representations into executable SQL queries.
8. To evaluate model performance using the official WikiSQL evaluator.
9. To visualize training and validation loss.
10. To develop a Streamlit frontend for interactive testing.
11. To analyze model limitations and improve the accuracy of SQL generation.

## 4. Dataset: WikiSQL

The project uses the **WikiSQL dataset**, which contains natural language questions paired with SQL queries and corresponding database table schemas.

The dataset is divided into three major splits:

| Dataset Split | Number of Examples |
|---|---:|
| Training | 56,355 |
| Validation | 8,421 |
| Testing | 15,878 |
| Total | 80,654 |

During training, 19 training examples were removed because they exceeded the configured maximum source or target sequence lengths. Therefore, the final training set contained 56,336 samples.

The dataset contains three important components:

### Natural Language Questions

These are questions written in ordinary English, such as:

- What is the nationality of the player?
- Which team had a pick of 24?
- What position does the player have?
- How many players are there?

### Table Schemas

Each question is associated with a database table containing column names. The schema is essential because the model must identify which columns are relevant to the question.

For example:

```text
Player
No.
Nationality
Position
Years in Toronto
School/Club Team
```

### SQL Query Representations

The original WikiSQL SQL queries contain information about selected columns, aggregation operations, and conditions.

The project converts these queries into a simplified textual format that can be learned by the Transformer.

For example:

```text
select <c2> where <c0> = terrence ross
```

Here, `<c2>` represents the third column, Nationality, and `<c0>` represents the first column, Player.

## 5. Data Preprocessing

The data preprocessing stage prepares the original WikiSQL dataset for training.

The preprocessing pipeline performs the following operations:

1. Loads the questions and table schemas from the WikiSQL files.
2. Associates each question with its corresponding table.
3. Extracts the table's column names.
4. Converts the column names into indexed schema tokens.
5. Converts the original SQL query into a simplified target representation.
6. Saves the processed data into JSON Lines files.
7. Creates a text corpus for training the tokenizer.

The source representation contains the natural language question followed by the table schema.

The source format was improved by introducing an explicit schema marker:

```text
question <sep> columns <c0> player <c1> no. <c2> nationality ...
```

The `<sep>` token separates the natural language question from the database schema, while the `columns` marker identifies the beginning of the schema information.

The target representation follows a simplified SQL format:

```text
select <c2> where <c0> = terrence ross
```

This representation allows the model to learn SQL structure without directly generating complete database-specific SQL syntax.

## 6. Tokenization

The project uses **SentencePiece BPE tokenization** with a vocabulary size of 8,000 tokens.

SentencePiece converts text into token IDs that can be processed by the Transformer.

The tokenizer includes special tokens:

| Token | ID | Purpose |
|---|---:|---|
| PAD | 0 | Padding |
| UNK | 1 | Unknown token |
| BOS | 2 | Beginning of sequence |
| EOS | 3 | End of sequence |
| `<sep>` | 4 | Separates question and schema |
| `<c0>` | 5 | First schema column |

Additional schema tokens are also included in the tokenizer vocabulary.

The source and target sequences are converted into numerical representations. Padding is applied so that examples in a batch have the same sequence length.

The training configuration uses:

- Maximum source length: 160 tokens
- Maximum target length: 64 tokens
- Batch size: 64

## 7. Transformer Architecture

The project implements an encoder-decoder Transformer architecture from scratch using PyTorch.

The architecture follows the configuration specified in the project requirements.

### Model Configuration

| Parameter | Value |
|---|---|
| Model Type | Encoder-Decoder Transformer |
| Embedding Dimension | 256 |
| Attention Heads | 4 |
| Encoder Layers | 3 |
| Decoder Layers | 3 |
| Feed-Forward Dimension | 1,024 |
| Dropout | 0.1 |
| Vocabulary Size | 8,000 |
| Batch Size | 64 |
| Training Epochs | 20 |
| Optimizer | Adam |
| Label Smoothing | 0.1 |
| Beam Search Size | 4 |
| Trainable Parameters | 7,577,600 |

The model contains approximately 7.58 million trainable parameters.

### 7.1 Token Embedding

Token embedding converts token IDs into dense vector representations.

Each token is represented by a 256-dimensional vector. These embeddings are learned during training.

The embedding vectors allow the model to represent relationships between natural language words, schema tokens, and SQL operations.

### 7.2 Positional Encoding

Since the Transformer does not use recurrence, positional encoding is added to token embeddings to provide information about the order of tokens.

The project uses sinusoidal positional encoding based on sine and cosine functions.

The positional encoding is calculated using:

\[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

where:

- \(pos\) represents the token position.
- \(i\) represents the embedding dimension index.
- \(d_{model}=256\).

A positional encoding heatmap was generated to visualize how positional information varies across token positions and embedding dimensions.

### 7.3 Scaled Dot-Product Attention

Scaled dot-product attention calculates the relevance of one token to other tokens.

The attention operation is defined as:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where:

- \(Q\) represents queries.
- \(K\) represents keys.
- \(V\) represents values.
- \(d_k\) is the key dimension.

The attention mechanism enables the model to focus on relevant parts of the question and schema.

### 7.4 Multi-Head Attention

Multi-head attention allows the model to learn different relationships between tokens simultaneously.

The model uses four attention heads. Since the embedding dimension is 256, each attention head operates on a dimension of:

\[
d_k=\frac{256}{4}=64
\]

The attention heads are combined and passed through a linear projection.

### 7.5 Feed-Forward Network

Each encoder and decoder layer contains a position-wise feed-forward network.

The network consists of:

- A linear layer from 256 to 1,024 dimensions.
- ReLU activation.
- Dropout.
- A linear layer from 1,024 back to 256 dimensions.

This component processes the information extracted by the attention mechanism.

### 7.6 Encoder

The encoder contains three encoder layers.

Each encoder layer includes:

1. Multi-head self-attention.
2. Residual connection.
3. Layer normalization.
4. Feed-forward network.
5. Another residual connection and layer normalization.

The encoder processes the natural language question and database schema to generate contextual representations.

### 7.7 Decoder

The decoder contains three decoder layers.

Each decoder layer includes:

1. Masked multi-head self-attention.
2. Encoder-decoder cross-attention.
3. Feed-forward network.
4. Residual connections.
5. Layer normalization.

The decoder generates the target SQL sequence token by token.

During training, the decoder receives the target sequence shifted by one position. During inference, it generates tokens sequentially.

### 7.8 Padding and Causal Masks

The project implements two types of masks.

**Padding mask:** Prevents the model from attending to padding tokens.

**Causal mask:** Prevents the decoder from attending to future tokens during sequence generation.

These masks ensure that the Transformer learns and generates sequences correctly.

## 8. Model Training

The Transformer was trained using the WikiSQL training set and validated using the validation set.

The training process follows these steps:

1. Load a batch of source and target sequences.
2. Move the batch to the available computing device.
3. Shift the target sequence to create decoder input and expected output.
4. Generate predictions using the Transformer.
5. Calculate cross-entropy loss.
6. Apply label smoothing.
7. Perform backpropagation.
8. Clip gradients to prevent excessively large updates.
9. Update the model parameters using Adam.
10. Evaluate the model on the validation set.
11. Save the best model checkpoint based on validation loss.

The model was trained on an NVIDIA GeForce RTX 4050 Laptop GPU using CUDA.

### Training Loss

The training loss decreased throughout the 20 epochs.

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 8.1425 | 4.1233 |
| 5 | 2.7434 | 2.7088 |
| 10 | 2.3393 | 2.3902 |
| 15 | 2.1069 | 2.2110 |
| 20 | 1.9726 | 2.1846 |

The best validation loss recorded during the training run was approximately 2.1846 at epoch 20.

The decreasing training and validation losses indicate that the model learned the general structure of the target SQL representation.

The best model checkpoint was saved as:

```text
best_model.pt
```

## 9. SQL Generation

After training, the model generates SQL representations from natural language questions and table schemas.

The project implements two decoding methods.

### 9.1 Greedy Decoding

Greedy decoding selects the token with the highest probability at every generation step.

The process continues until the model generates the EOS token or reaches the maximum sequence length.

Greedy decoding is computationally efficient, but it may produce suboptimal sequences because it only considers the most likely token at each step.

### 9.2 Beam Search

Beam search explores multiple candidate sequences during generation.

The project uses a beam size of four.

At each step, the decoder considers the top candidate tokens and retains the four highest-scoring sequences.

Beam search can produce better sequences than greedy decoding because it considers multiple possible generation paths.

However, beam search does not guarantee that the generated SQL query is correct. It depends on the probabilities learned by the trained model.

## 10. SQL Conversion

The generated model output is a simplified SQL representation containing indexed column tokens.

For example:

```text
select <c2> where <c0> = terrence ross
```

The SQL converter maps the indexed column tokens to the actual table column names.

Given the following schema:

```text
Player, No., Nationality, Position, Years in Toronto, School/Club Team
```

The converter produces:

```sql
SELECT Nationality
WHERE Player = 'terrence ross'
```

The converter also handles aggregation operations such as:

- MAX
- MIN
- COUNT
- SUM
- AVG

The generated query is then prepared for evaluation using the official WikiSQL evaluator.

## 11. Evaluation

The project uses the official WikiSQL evaluator to measure the model's performance.

Two evaluation metrics are used.

### 11.1 Execution Accuracy

Execution accuracy measures the percentage of generated queries that return the same execution result as the ground-truth query.

This metric evaluates whether the generated SQL query retrieves the correct result from the database.

### 11.2 Logical Form Accuracy

Logical form accuracy measures the percentage of generated queries that match the ground-truth SQL structure.

This metric evaluates the structural correctness of the generated SQL representation.

### Initial Model Results

The first completed training run produced the following results:

| Metric | Initial Result |
|---|---:|
| Execution Accuracy | 14.69% |
| Logical Form Accuracy | 7.83% |

These results indicated that the initial model learned parts of the SQL generation process but struggled to select the correct columns and conditions consistently.

### Source Representation Improvement

An improvement was introduced in the source representation by explicitly marking the beginning of the schema:

```text
question <sep> columns <c0> player <c1> no. ...
```

The tokenizer was regenerated, and the model was retrained using the updated representation.

During testing, the updated model generated:

```text
select <c4> where <c5> = terrence ross
```

and, using beam search:

```text
select <c5> where <c4> = terrence ross
```

The expected structure for the question:

> What school or club team did Terrence Ross play for?

is:

```text
select <c5> where <c0> = terrence ross
```

The updated model showed improvement in identifying the requested output column in the beam-search result, but it still produced an incorrect condition column.

**The official test accuracy of the retrained model has not yet been confirmed.** The updated predictions need to be evaluated before reporting final performance.

## 12. Streamlit Frontend

A Streamlit frontend was developed to provide an interactive interface for the Text-to-SQL Transformer.

The frontend allows users to:

1. Enter a natural language question.
2. Provide the table's column names.
3. Generate an SQL query.
4. View the generated SQL.
5. View the model's intermediate output.
6. Review the training and validation loss graph.
7. Review the model's evaluation results.
8. View the model configuration.

The application loads the trained Transformer checkpoint and performs inference using the available GPU.

The application can be launched using:

```powershell
streamlit run app.py
```

The frontend is designed to demonstrate the complete workflow from natural language input to generated SQL output.

## 13. Results Visualization

The project includes a training and validation loss graph.

The graph displays the loss values across the 20 training epochs.

The visualization helps analyze:

- How quickly the model learns.
- Whether training loss decreases.
- Whether validation loss improves.
- Whether the model begins to overfit.
- How training and validation performance compare.

The graph is generated using Matplotlib and is also displayed inside the Streamlit application.

The results are stored in JSON format, allowing the frontend to display model metrics and configuration values dynamically.

## 14. Technologies Used

The project uses the following technologies and libraries:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Transformer implementation and training |
| CUDA | GPU acceleration |
| SentencePiece | BPE tokenization |
| WikiSQL | Text-to-SQL dataset |
| NumPy | Numerical operations |
| Matplotlib | Training visualizations |
| Streamlit | Interactive frontend |
| JSON / JSONL | Dataset and result storage |
| Official WikiSQL Evaluator | Model evaluation |

## 15. Challenges and Limitations

Several challenges were encountered during development.

### Column Selection

The model sometimes selects an incorrect output column even when the question is understandable.

For example, a question asking for a player's school or club team may result in the model selecting the Position column.

### Condition Selection

The model may correctly identify the output column but incorrectly select the filtering column.

For example, it may generate:

```text
select <c5> where <c4> = terrence ross
```

instead of:

```text
select <c5> where <c0> = terrence ross
```

### Generalization

The model must generalize to questions and database schemas that it has not encountered during training.

A question may use wording that differs from the examples in the training dataset.

### SQL Structure

The model must generate the correct sequence of SQL components, including SELECT, WHERE, aggregation operations, and conditions.

An error in any of these components can result in an incorrect query.

### Accuracy

The initial official evaluation results were relatively low. Therefore, further improvements to the training process, source representation
