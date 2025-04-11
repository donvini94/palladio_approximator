{
  description = "Palladio Performance Prediction MVP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        python = pkgs.python312;
        pythonPackages = python.withPackages (
          ps: with ps; [
            torch
            scikit-learn
            pandas
            tqdm
          ]
        );
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonPackages
            pkgs.poetry
          ];

          shellHook = ''
            export POETRY_VIRTUALENVS_IN_PROJECT=true

            if [ ! -f "pyproject.toml" ]; then
              poetry init --no-interaction --name dsl_predictor
              echo "Poetry project initialized with pyproject.toml"
            fi

            echo "✅ Environment ready. You can run: python train.py --root_dir path/to/top_level_directory/"
          '';
        };
      }
    );
}
