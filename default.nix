{ lib, config, pkgs, ... }: let

  boneless = pkgs.runCommand "boneless" {
    src = pkgs.fetchurl {
      url = "https://delivery.r8.co/sdxl-nyacomp4/boneless_model.pth";
      hash = "sha256-M5/uReNxSK0iZ5CpFUw2uKpzNusbgqgQbHS6LMw/YIg=";
    };
  } ''mkdir -p $out/src && cp $src $out/src/boneless_model.pth'';

  # remotefile = pkgs.runCommand "remotefile" {
  #   src = pkgs.fetchurl {
  #     url = "https://r2.drysys.workers.dev/remotefile";
  #     hash = "sha256-LlOC3tTm0HMqITqDOGjLo8tXA/qTogh821Dcaorzk0I=";
  #   };
  # } ''install -Dm755 $src $out/usr/local/bin/remotefile'';

  # pget = pkgs.runCommand "pget" {
  #   src = pkgs.fetchurl {
  #     url = "https://github.com/replicate/pget/releases/download/v0.5.6/pget_linux_x86_64";
  #     hash = "sha256-2lbonESSQ/pXDp2QoQo5nhIydk3RlCMRAMi54T8vRoI=";
  #   };
  # } ''install -Dm755 $src $out/bin/pget'';
in {
  # just ignore the run things
  options.cog.build.run = lib.mkOption {};
  config = {
    cog.build = {
      cog_version = "0.9.5";
      # override all of the system packages
      system_packages = lib.mkForce [ "ffmpeg-headless" boneless ];
    };
    python-env.pip.drvs = let self = config.python-env.pip.drvs; in {
      # sigh... torch includes libcudart-d0da41ae.so.11.0, who's loading that?
      torchvision.env.autoPatchelfIgnoreMissingDeps = [ "libcudart.so.11.0" ];
      nyacomp.env.autoPatchelfIgnoreMissingDeps = [ "libcudart.so.11.0" ];
      # inject torch dep
      nyacomp.mkDerivation.propagatedBuildInputs = [ self.torch.public ];
    };
  };
}
