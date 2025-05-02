{
  description = "Python development environment with CUDA-enabled PyTorch";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      pythonEnv = pkgs.python312.withPackages (
        ps: with ps; [
          scikit-learn
          pandas
          tqdm
          numpy
          joblib
          transformers
          pytorchWithCuda
          seaborn
          tiktoken # some embedding models need it
        ]
      );

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pythonEnv
          pkgs.uv
        ];

        shellHook = ''
          export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
          echo "Torch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

          # Create uv environment in .venv if it doesn't exist
          if [ ! -d .venv ]; then
            echo "[INFO] Creating .venv using uv..."
            uv venv .venv
          fi

          # Activate it
          source .venv/bin/activate

          # Sync dependencies if needed
          if [ -f requirements.txt ]; then
            uv pip install -r requirements.txt
          fi
        '';
      };
    };
}
