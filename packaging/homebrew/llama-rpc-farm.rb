cask "llama-rpc-farm" do
  version "2.0.52"
  sha256 :no_check

  url "https://github.com/dakotacookson/Llama-RPC-Farm/releases/download/release-#{version}/llama-rpc-farm_#{version}_macos-arm64.pkg"
  name "Llama RPC Farm"
  desc "Pool home machines into one llama.cpp RPC cluster"
  homepage "https://github.com/dakotacookson/Llama-RPC-Farm"

  depends_on macos: ">= :ventura"
  depends_on arch: :arm64

  pkg "llama-rpc-farm_macos-arm64.pkg"

  uninstall pkgutil: "com.dakotacookson.llamarpcfarm"

  zap trash: [
    "~/Library/Application Support/LlamaRPCFarm",
    "~/.llama-rpc-farm",
    "~/.llama-peer-agent",
  ]
end
