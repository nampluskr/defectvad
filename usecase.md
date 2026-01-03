# Use Cases

### 모델 생성

```python
model_config = load_config(model_dir, "config.yaml", "model")
model = create_model(model_config)
```

```python
trainer_config = load_config(model_dir, "config.yaml", "trainer)
trainer = create_trainer(model, trainer_config)
trainer.fit(train_loader)
```