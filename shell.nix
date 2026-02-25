{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    cmake
    gnumake
    gcc
  ];

  buildInputs = with pkgs; [
    openblas
    gflags 
    python3
    python3Packages.numpy
    python3Packages.matplotlib
    python3Packages.pandas
  ];

}