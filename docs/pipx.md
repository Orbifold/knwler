# Pipx Setup

You can run Knwler in its own isolated environment using [pipx](https://pipx.pypa.io/stable/installation/), it means all dependencies are installed but not shared. It's the closes things to having a python package as an application (on Mac, Windows and Linux).

Check the pipx pages for details and once you have it installed you can setup Knwler with

```bash
pipx install knwler --python python3.12
```
Now you can use Knwler directly without invoking `uv`:

```bash
knwler --help
```

An additional advantage of using pipx is that you can upgrade directly with

```bash
pipx ugrade knwler
```
