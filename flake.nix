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
        ]
      );

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pythonEnv
        ];

        shellHook = ''
          echo "Torch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
        '';
      };
    };
}
