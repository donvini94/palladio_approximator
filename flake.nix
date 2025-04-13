{
  description = "Python development environment with Poetry";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs =
    { self, nixpkgs }:
    {
      devShells = {
        x86_64-linux.default =
          let
            pkgs = import nixpkgs { system = "x86_64-linux"; };
            python = pkgs.python312;
            pythonPackages = python.withPackages (
              ps: with ps; [
                torch
                scikit-learn
                pandas
                tqdm
                numpy
                joblib
              ]
            );
          in
          pkgs.mkShell {
            buildInputs = [
              pkgs.python312
              pkgs.poetry
              pythonPackages
            ];
            shellHook = ''
              # Ensure Poetry uses the virtual environment in the project folder
              export POETRY_VIRTUALENVS_IN_PROJECT=true

              # Initialize Poetry if not already done
              if [ ! -f "pyproject.toml" ]; then
                poetry init --no-interaction --name tcpm_generator
                echo "Poetry project initialized with pyproject.toml"
              fi
            '';
          };
      };
    };
}
