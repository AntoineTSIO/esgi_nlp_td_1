import click
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from data import make_dataset
from feature import make_features, preprocess_text
from models import create_model


@click.group()
def cli():
    """Command line interface for the NLP classification project."""
    pass


@click.command()
@click.option(
    "--input_filename", default="data/raw/train.csv", help="File containing training data"
)
@click.option(
    "--model_dump_filename", default="models/dump.json", help="File to save the trained model"
)
@click.option("--model_type", default="rf", help="Type of model to use (rf or lr)")
def train(input_filename, model_dump_filename, model_type):
    """Train a model and save it to a file."""
    dataset = make_dataset(input_filename)
    features, labels, vectorizer = make_features(dataset)

    model = create_model(model_type)
    model.fit(features, labels)

    joblib.dump((model, vectorizer), model_dump_filename)
    click.echo(f"Model trained and saved to {model_dump_filename}")


@click.command()
@click.option(
    "--input_filename", default="data/raw/test.csv", help="File with data to predict"
)
@click.option(
    "--model_dump_filename", default="models/dump.json", help="File with dumped model"
)
@click.option(
    "--output_filename",
    default="data/processed/prediction.csv",
    help="Output file for predictions",
)
def predict(input_filename, model_dump_filename, output_filename):
    """Make predictions using a trained model."""
    dataset = make_dataset(input_filename)
    model, vectorizer = joblib.load(model_dump_filename)

    dataset["processed_title"] = dataset["video_name"].apply(preprocess_text)
    features = vectorizer.transform(dataset["processed_title"])

    predictions = model.predict(features)

    dataset["predicted_is_comic"] = predictions
    dataset.to_csv(output_filename, index=False)
    click.echo(f"Predictions saved to {output_filename}")


@click.command()
@click.option(
    "--input_filename", default="data/raw/train.csv", help="File containing training data"
)
@click.option("--model_type", default="rf", help="Type of model to use (rf or lr)")
def evaluate(input_filename, model_type):
    """Evaluate the model using cross-validation."""
    dataset = make_dataset(input_filename)
    features, labels, vectorizer = make_features(dataset)
    model = create_model(model_type)

    scores = cross_val_score(model, features, labels, cv=5, scoring="accuracy")
    click.echo(f"Cross-validation scores: {scores}")
    click.echo(f"Mean accuracy: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")


@click.command()
@click.option(
    "--input_filename", default="data/raw/dataset.csv", help="File with full dataset"
)
@click.option(
    "--train_filename", default="data/raw/train.csv", help="Output file for training data"
)
@click.option(
    "--test_filename", default="data/raw/test.csv", help="Output file for test data"
)
@click.option(
    "--test_size",
    default=0.2,
    help="Proportion of the dataset to include in the test split",
)
def split_dataset(input_filename, train_filename, test_filename, test_size):
    """Split the dataset into training and test sets."""
    dataset = make_dataset(input_filename)
    train_df, test_df = train_test_split(dataset, test_size=test_size, random_state=42)

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)

    click.echo(f"Dataset split into {train_filename} and {test_filename}")


@click.command()
def help():
    """Display help information for all commands."""
    ctx = click.get_current_context()
    if ctx.parent is not None:
        click.echo(ctx.parent.get_help())
    else:
        click.echo(ctx.get_help())
    click.echo("\nAvailable commands:")
    commands = cli.commands.keys()
    for cmd in commands:
        if cmd != "help":
            click.echo(f"  {cmd}: {cli.commands[cmd].__doc__}")


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(split_dataset)
cli.add_command(help)


if __name__ == "__main__":
    cli()
