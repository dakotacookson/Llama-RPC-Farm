# Examples

Minimal examples with step by step procedures.

## Single-device Smoke test (MacOS Metal)

Run a small GGUF model locally on the MacOS

### Prerequisites

| Component | Requirement |
|---|---|
| Mac | Apple Silicon; llama-server built with Metal and RPC support |


### Get the model

Model-download placeholder: obtain `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` and place it in `$HOME/models`.

```bash
./get_model.sh
```
![alt text](img/get_dummy_model.png)

### Terminal A — inference server

```bash
./run_mac_local.sh 
```

Wait until the model finishes loading. The HTTP endpoint is `http://127.0.0.1:8084`.

![alt text](img/run_mac_local.png)

Larger models might take longer to load.

### Sample generation

With Terminal A running the service, run:

```bash
./test_inference.sh "<SAMPLE_TEXT>" <MAX_OUTPUT_TOKENS>
```


Example:

```bash
./test_inference.sh "The capital of Austria is" 32
```

![alt text](img/test_inference_single.png)


## Server Smoke test (MacOS Metal + Linux RAM CPU)

Run a small GGUF model on a Mac with a Linux RPC worker. 

### Prerequisites

| Component | Requirement |
|---|---|
| Mac (Master) | Apple Silicon; llama-server built with Metal and RPC support |
| Linux (worker) | Compatible ggml-rpc-server build; CPU device available |
| Network  | For SSH **password** over **WLAN**: SSH running; TCP port 50052 reachable; **Trusted VPN might be needed** (ports in router disabled by default) |
| Network  | For SSH **password** over **Ethernet**: SSH running; TCP port 50052 reachable; |
| Network  | *For cases SSH **Key** (any: WLAN/Eth):* SSH running; TCP port 50052 reachable; |
| Ethernet Connection (For SSH password connections) | SSH  running; TCP port 50052 reachable|
| Worker binary | Verify its Linux path; default is `$HOME/llama-rpc-worker/build/bin/ggml-rpc-server` |

> Run the commands below from the repository's `scripts` directory. Replace `<YOUR_IP>` and `<YOUR_USER_NAME>` with your Linux worker's address and SSH username.

### Get the model

Model-download placeholder: obtain `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` and place it in `$HOME/models`.

```bash
./get_dummy_model.sh
```
![alt text](img/get_dummy_model.png)

### Terminal A — worker logs

```bash
./start_worker_linux.sh <YOUR_IP> <YOUR_USER_NAME>
```

The worker must already be running for the currently verified path; automatic startup has unresolved binary-path and log-redirection quoting issues.

![Worker startup and logs](img/start_worker_linux.png)

### Terminal B — inference server

```bash
./run_cli.sh <YOUR_IP> <YOUR_USER_NAME>
```

Wait until the model finishes loading. The HTTP endpoint is `http://127.0.0.1:8084`.

![Inference server](img/run_cli.png)

### Terminal C — sample generation

```bash
./test_inference.sh "<SAMPLE_TEXT>" <MAX_OUTPUT_TOKENS>
```

Example:

```bash
./test_inference.sh "The capital of Austria is" 32
```

![Running test](img/test_inference_init.png)

![Completed test](img/test_inference_end.png)

