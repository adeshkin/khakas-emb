import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from huggingface_hub import login


def main():
    login()
    data_dir = "/content/drive/MyDrive/labse"
    repo_name = "labse-kjh-ru-new123"

    df = pd.read_csv(f"{data_dir}/para_kjh_ru.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    dataset = Dataset.from_dict({
        "anchor": df["kjh"].tolist()[:1200],
        "positive": df["ru"].tolist()[:1200]
    })

    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    model_name = "cointegrated/LaBSE-en-ru"
    model = SentenceTransformer(model_name)

    train_loss = MultipleNegativesRankingLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir=f"{data_dir}/{repo_name}",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=0.1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        fp16=True,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss=train_loss,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
