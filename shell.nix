let
    pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/ed30f8aba416.tar.gz") {
        config = {
            enableCuda = true;
            allowUnfree = true;
        };

        overlays = [ # Modify Python library to have overrides
            (
                final: prev: rec {
                    python312 = prev.python312.override {
                        self = python312;
                        packageOverrides = final_: prev_: {
                          torch = final_.torch-bin.overrideAttrs(torch-binFinalAttrs: torch-binPrevAttrs: {
                            passthru = torch-binPrevAttrs.passthru // {
                              cudaPackages = pkgs.cudaPackages;
                              cudaSupport = true;
                              cudaCapabilities = ["7.5"];
                            };
                          });
                          torchvision = final_.torchvision-bin;
                          torchaudio = final_.torchaudio-bin;
                          trl = final_.callPackage ./build/trl/default.nix { };
                          sklearn-compat = final_.buildPythonPackage rec { # Use pre-compiled version of sklearn-compat
                            pname = "sklearn_compat";
                            version = "0.1.3";
                            format = "wheel";

                            src = final_.fetchPypi {
                                inherit pname version format;
                                sha256 = "sha256-qKr473EZiMvWPxh8VWC18Wsl32Y6qh0tDhKRNB0zn4A=";
                                dist = "py3";
                                python = "py3";
                            };

                            propagatedBuildInputs = with final_; [ scikit-learn ];
                            doCheck = false;
                          };
                          imbalanced-learn = prev_.imbalanced-learn.overridePythonAttrs(imbalPrevAttrs: {
                            dependencies = imbalPrevAttrs.dependencies ++ [ final_.sklearn-compat ];
                            doCheck = false;
                          });
                          cut-cross-entropy = prev_.cut-cross-entropy.overridePythonAttrs(prevAttrs: {
                            dependencies = pkgs.lib.filter (dep: dep != final_.triton) prevAttrs.dependencies;
                            doCheck = false;
                          });
                          tyro = prev_.tyro.overridePythonAttrs(prevAttrs: {
                            dependencies = pkgs.lib.filter (dep: dep != final_.triton) prevAttrs.dependencies;
                            doCheck = false;
                          });
                          unsloth = prev_.unsloth.overridePythonAttrs(prevAttrs: {
                            dependencies = pkgs.lib.filter (dep: dep != final_.triton) prevAttrs.dependencies;
                            doCheck = false;
                          });
                          unsloth-zoo = prev_.unsloth-zoo.overridePythonAttrs(prevAttrs: {
                            dependencies = pkgs.lib.filter (dep: dep != final_.triton) prevAttrs.dependencies ++ [final_.pillow final_.protobuf];
                            doCheck = false;
                          });
                       };
                    };
                }
            )
        ];
};
in
pkgs.mkShell {
    buildInputs = with pkgs; [
        discordchatexporter-cli
        tmux
        lunarvim
        gh
        pipreqs
        (python312.withPackages (p: with p; [
            ipykernel
            jupyter
            pip
            numpy
            pandas
            torch
            torchvision
            torchaudio
            tqdm
            matplotlib
            bitsandbytes
            transformers
            peft
            accelerate
            datasets
            trl
            scikit-learn
            pytest
            openpyxl
            xlrd
            imbalanced-learn
            litellm
            tyro
            cut-cross-entropy
            unsloth
            unsloth-zoo
            pillow
            protobuf
            discordpy
            python-dotenv
            pip-tools
#             scipy
#             einops
#             evaluate
#             rouge_score
        ]
        ))
    ];
}
