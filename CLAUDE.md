# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- **Setup**: `nix develop` or `pip install -r requirements.txt` or `./run.sh build`
- **Run**: `python train.py --model [rf|torch|linear] --embedding [tfidf|bert] --prediction_mode [summary|detail]`
- **Docker**: `./run.sh train rf bert summary`
- **Monitor**: `./run.sh mlflow` - View experiment tracking

## Code Style
- **Indentation**: 4 spaces
- **Line length**: ~88-100 characters
- **Quotes**: Double quotes for strings
- **Imports**: Group by stdlib → third-party → local modules
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Documentation**: Google-style docstrings with Args/Returns sections
- **Error handling**: Use specific exception types with informative messages
- **Types**: Limited type annotations, use when adding new functions

## Project Structure
- **models/**: Different model implementations 
- **utils/**: Helper functions and visualization tools
- **data/**: Dataset storage
- Modular organization with separate responsibilities per file