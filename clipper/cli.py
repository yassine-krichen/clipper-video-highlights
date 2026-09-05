import typer

app = typer.Typer()


@app.callback()
def main() -> None:
    """clipper: long-form video to short-form clips."""


@app.command()
def version() -> None:
    """print the installed clipper version."""
    print("clipper 0.1.0")


if __name__ == "__main__":
    app()
