# Module & One-Line Command Cheat Sheet

Fast lookup for "what's the exact import and the one-line call." Same pattern throughout:
**for train/test split — it is `sklearn.model_selection.train_test_split` . `train_test_split(X, y, random_state=42, test_size=0.2)`**

94 entries across ML, Neural Networks, CNN, RNN/LSTM, LLM, Optimization, Training, and Inferencing.

## Machine Learning (scikit-learn)

| Task | Module path | One-line call |
|---|---|---|
| Train/test split | `sklearn.model_selection.train_test_split` | `train_test_split(X, y, random_state=42, test_size=0.2)` |
| Standardize features (zero mean, unit variance) | `sklearn.preprocessing.StandardScaler` | `StandardScaler().fit_transform(X)` |
| Min-max scale to [0,1] | `sklearn.preprocessing.MinMaxScaler` | `MinMaxScaler().fit_transform(X)` |
| Encode labels as integers | `sklearn.preprocessing.LabelEncoder` | `LabelEncoder().fit_transform(y)` |
| One-hot encode categoricals | `sklearn.preprocessing.OneHotEncoder` | `OneHotEncoder(sparse_output=False).fit_transform(X)` |
| Fill missing values | `sklearn.impute.SimpleImputer` | `SimpleImputer(strategy="mean").fit_transform(X)` |
| Chain preprocessing + model | `sklearn.pipeline.Pipeline` | `Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])` |
| Different transforms per column | `sklearn.compose.ColumnTransformer` | `ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(), cat_cols)])` |
| K-fold cross-validation score | `sklearn.model_selection.cross_val_score` | `cross_val_score(model, X, y, cv=5)` |
| Grid search hyperparameters | `sklearn.model_selection.GridSearchCV` | `GridSearchCV(model, param_grid, cv=5).fit(X, y)` |
| Random search hyperparameters | `sklearn.model_selection.RandomizedSearchCV` | `RandomizedSearchCV(model, param_dist, n_iter=20).fit(X, y)` |
| Logistic regression | `sklearn.linear_model.LogisticRegression` | `LogisticRegression(max_iter=1000).fit(X_train, y_train)` |
| Linear regression | `sklearn.linear_model.LinearRegression` | `LinearRegression().fit(X_train, y_train)` |
| Random forest classifier | `sklearn.ensemble.RandomForestClassifier` | `RandomForestClassifier(n_estimators=200).fit(X_train, y_train)` |
| Gradient boosting classifier | `sklearn.ensemble.GradientBoostingClassifier` | `GradientBoostingClassifier().fit(X_train, y_train)` |
| Support vector classifier | `sklearn.svm.SVC` | `SVC(kernel="rbf", C=1.0).fit(X_train, y_train)` |
| K-nearest neighbors | `sklearn.neighbors.KNeighborsClassifier` | `KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)` |
| K-means clustering | `sklearn.cluster.KMeans` | `KMeans(n_clusters=3, n_init="auto").fit(X)` |
| PCA dimensionality reduction | `sklearn.decomposition.PCA` | `PCA(n_components=2).fit_transform(X)` |
| Confusion matrix | `sklearn.metrics.confusion_matrix` | `confusion_matrix(y_true, y_pred)` |
| Precision/recall/F1 report | `sklearn.metrics.classification_report` | `classification_report(y_true, y_pred)` |
| ROC-AUC score | `sklearn.metrics.roc_auc_score` | `roc_auc_score(y_true, y_score)` |
| Mean squared error | `sklearn.metrics.mean_squared_error` | `mean_squared_error(y_true, y_pred)` |
| R² score | `sklearn.metrics.r2_score` | `r2_score(y_true, y_pred)` |
| Save a fitted model | `joblib` | `joblib.dump(model, "model.joblib")` |
| Load a fitted model | `joblib` | `joblib.load("model.joblib")` |
| Oversample a minority class | `imblearn.over_sampling.SMOTE` | `SMOTE(random_state=42).fit_resample(X, y)` |

## Neural Networks (PyTorch core)

| Task | Module path | One-line call |
|---|---|---|
| Fully-connected layer | `torch.nn.Linear` | `nn.Linear(in_features=768, out_features=256)` |
| Base class for any model | `torch.nn.Module` | `class Net(nn.Module): ...` (implement `__init__` + `forward`) |
| ReLU activation | `torch.nn.ReLU` | `nn.ReLU()` |
| Dropout regularization | `torch.nn.Dropout` | `nn.Dropout(p=0.3)` |
| Batch normalization | `torch.nn.BatchNorm1d` | `nn.BatchNorm1d(num_features=256)` |
| Layer normalization | `torch.nn.LayerNorm` | `nn.LayerNorm(normalized_shape=768)` |
| Cross-entropy loss | `torch.nn.CrossEntropyLoss` | `nn.CrossEntropyLoss()(logits, targets)` |
| Mean squared error loss | `torch.nn.MSELoss` | `nn.MSELoss()(pred, target)` |
| Batch and shuffle a dataset | `torch.utils.data.DataLoader` | `DataLoader(dataset, batch_size=32, shuffle=True)` |
| Custom dataset class | `torch.utils.data.Dataset` | `class MyData(Dataset): ...` (implement `__len__` + `__getitem__`) |
| Move model/tensor to GPU | `torch.Tensor.to` | `model.to("cuda")` |
| Save model weights | `torch.save` | `torch.save(model.state_dict(), "model.pt")` |
| Load model weights | `torch.load` | `model.load_state_dict(torch.load("model.pt"))` |
| Switch to eval mode (disables dropout) | `torch.nn.Module.eval` | `model.eval()` |
| Switch to train mode | `torch.nn.Module.train` | `model.train()` |
| Disable gradient tracking (inference) | `torch.no_grad` | `with torch.no_grad(): preds = model(x)` |

## CNN

| Task | Module path | One-line call |
|---|---|---|
| 2D convolution layer | `torch.nn.Conv2d` | `nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)` |
| Max pooling | `torch.nn.MaxPool2d` | `nn.MaxPool2d(kernel_size=2, stride=2)` |
| Adaptive average pooling (any input size → fixed output) | `torch.nn.AdaptiveAvgPool2d` | `nn.AdaptiveAvgPool2d(output_size=(1, 1))` |
| Image augmentation pipeline | `torchvision.transforms` | `transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])` |
| Pretrained CNN backbone | `torchvision.models` | `models.resnet18(weights="IMAGENET1K_V1")` |

## RNN & LSTM

| Task | Module path | One-line call |
|---|---|---|
| Vanilla RNN layer | `torch.nn.RNN` | `nn.RNN(input_size=100, hidden_size=128, batch_first=True)` |
| GRU layer | `torch.nn.GRU` | `nn.GRU(input_size=100, hidden_size=128, batch_first=True)` |
| LSTM layer | `torch.nn.LSTM` | `nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)` |
| Pack variable-length sequences | `torch.nn.utils.rnn.pack_padded_sequence` | `pack_padded_sequence(x, lengths, batch_first=True)` |
| Pad a batch of sequences to equal length | `torch.nn.utils.rnn.pad_sequence` | `pad_sequence(sequences, batch_first=True)` |

## LLM

| Task | Module path | One-line call |
|---|---|---|
| Load a tokenizer | `transformers.AutoTokenizer` | `AutoTokenizer.from_pretrained("gpt2")` |
| Load a causal (generative) LM | `transformers.AutoModelForCausalLM` | `AutoModelForCausalLM.from_pretrained("gpt2")` |
| Generate text from a prompt | `transformers.GenerationMixin.generate` | `model.generate(**inputs, max_new_tokens=50, do_sample=True, top_p=0.9)` |
| One-line text-generation pipeline | `transformers.pipeline` | `pipeline("text-generation", model="gpt2")("Once upon a time")` |
| Config a 4-bit quantized load (QLoRA) | `transformers.BitsAndBytesConfig` | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` |
| LoRA adapter config | `peft.LoraConfig` | `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])` |
| Attach a LoRA adapter to a model | `peft.get_peft_model` | `get_peft_model(model, lora_config)` |
| Supervised fine-tune trainer | `trl.SFTTrainer` | `SFTTrainer(model=model, train_dataset=ds, args=sft_config)` |
| Build a chat prompt template | `langchain_core.prompts.ChatPromptTemplate` | `ChatPromptTemplate.from_messages([("system", s), ("human", "{q}")])` |
| Call an Azure/OpenAI chat model | `langchain_openai.AzureChatOpenAI` | `AzureChatOpenAI(deployment_name="gpt-4o").invoke(messages)` |
| Embed text into vectors | `sentence_transformers.SentenceTransformer` | `SentenceTransformer("all-MiniLM-L6-v2").encode(sentences)` |
| Split text into chunks for RAG | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(doc)` |
| Build a vector index | `langchain_community.vectorstores.FAISS` | `FAISS.from_documents(docs, embedding_model)` |
| Query a vector index (top-k retrieve) | `langchain_community.vectorstores.FAISS` | `vectorstore.similarity_search(query, k=4)` |
| Define an agent state graph | `langgraph.graph.StateGraph` | `StateGraph(AgentState).add_node("llm", call_model)` |

## Optimization

| Task | Module path | One-line call |
|---|---|---|
| SGD optimizer | `torch.optim.SGD` | `optim.SGD(model.parameters(), lr=0.01, momentum=0.9)` |
| Adam optimizer | `torch.optim.Adam` | `optim.Adam(model.parameters(), lr=1e-3)` |
| AdamW optimizer (decoupled weight decay) | `torch.optim.AdamW` | `optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)` |
| Step-decay learning rate | `torch.optim.lr_scheduler.StepLR` | `StepLR(optimizer, step_size=10, gamma=0.1)` |
| Cosine annealing schedule | `torch.optim.lr_scheduler.CosineAnnealingLR` | `CosineAnnealingLR(optimizer, T_max=50)` |
| Linear warmup schedule (transformers) | `transformers.get_linear_schedule_with_warmup` | `get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)` |
| Clip exploding gradients | `torch.nn.utils.clip_grad_norm_` | `clip_grad_norm_(model.parameters(), max_norm=1.0)` |

## Training

| Task | Module path | One-line call |
|---|---|---|
| Zero out old gradients | `torch.optim.Optimizer.zero_grad` | `optimizer.zero_grad()` |
| Backpropagate the loss | `torch.Tensor.backward` | `loss.backward()` |
| Apply the gradient step | `torch.optim.Optimizer.step` | `optimizer.step()` |
| HF training config | `transformers.TrainingArguments` | `TrainingArguments(output_dir="out", per_device_train_batch_size=8, num_train_epochs=3)` |
| HF high-level trainer | `transformers.Trainer` | `Trainer(model=model, args=args, train_dataset=ds).train()` |
| Keras compile + fit | `tf.keras.Model` | `model.compile(optimizer="adam", loss="categorical_crossentropy"); model.fit(X, y, epochs=10)` |
| Stop training on plateau | `tf.keras.callbacks.EarlyStopping` | `EarlyStopping(monitor="val_loss", patience=3)` |
| Save the best checkpoint | `tf.keras.callbacks.ModelCheckpoint` | `ModelCheckpoint("best.h5", save_best_only=True)` |
| Multi-GPU/distributed helper | `accelerate.Accelerator` | `accelerator = Accelerator(); model, optimizer, dl = accelerator.prepare(model, optimizer, dl)` |
| Init a distributed process group | `torch.distributed.init_process_group` | `init_process_group(backend="nccl")` |
| Wrap a model for DDP | `torch.nn.parallel.DistributedDataParallel` | `DistributedDataParallel(model, device_ids=[rank])` |
| Mixed-precision autocast | `torch.autocast` | `with torch.autocast("cuda", dtype=torch.bfloat16): out = model(x)` |
| Gradient scaler for FP16 training | `torch.cuda.amp.GradScaler` | `scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()` |

## Inferencing

| Task | Module path | One-line call |
|---|---|---|
| High-throughput LLM batch serving | `vllm.LLM` | `LLM(model="meta-llama/Llama-2-7b-hf").generate(prompts, sampling_params)` |
| Decode token ids back to text | `transformers.PreTrainedTokenizer.decode` | `tokenizer.decode(output_ids, skip_special_tokens=True)` |
| Run an exported ONNX model | `onnxruntime.InferenceSession` | `InferenceSession("model.onnx").run(None, {"input": x})` |
| Export a model to ONNX | `torch.onnx.export` | `torch.onnx.export(model, dummy_input, "model.onnx")` |
| Compile to TorchScript for fast inference | `torch.jit.trace` | `torch.jit.trace(model, example_input)` |
| Pin which GPU is visible | `os.environ` | `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` |

## Practice Q&A (Self-Test)

### Which sklearn call splits data with a reproducible shuffle?
`train_test_split(X, y, random_state=42, test_size=0.2)` — `random_state` fixes the shuffle so results are reproducible; `test_size=0.2` holds out 20%.

### `model.eval()` vs `torch.no_grad()` — what does each actually turn off?
`model.eval()` flips layer *behavior* (Dropout stops dropping, BatchNorm uses running stats instead of batch stats). `torch.no_grad()` stops the *autograd engine* from building a graph, saving memory/compute. Use both together for inference — one without the other is a common bug (e.g. `eval()` without `no_grad()` still tracks gradients and wastes memory).

### Why does AdamW take a separate `weight_decay` argument instead of adding L2 to the loss?
Plain Adam+L2 folds the L2 penalty into the gradient, so it gets divided by Adam's per-parameter adaptive learning rate — parameters with large gradient history get decayed less. `AdamW` decouples weight decay from the gradient update so every parameter shrinks by the same proportion regardless of its gradient history — the standard optimizer for transformer fine-tuning.

### What's the one-line difference between a plain `AutoModelForCausalLM.from_pretrained(...)` load and a QLoRA load?
Add `quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...)` to the `from_pretrained(...)` call — same function, one extra kwarg, and the weights load already quantized to 4-bit.
