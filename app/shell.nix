with import <nixpkgs> { };
mkShell {
  nativeBuildInputs = with pkgs;
    [
      nodejs
      nodePackages.npm
      supabase-cli
      railway
      nixpacks
    ];
}
