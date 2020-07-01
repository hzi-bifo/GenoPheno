# GenoPheno

docker build --tag local/genopheno:0.95 .
docker run -v data:/app/GenoPheno/data -v output:/app/GenoPheno/output -v configs:/app/GenoPheno/configs -v config.yaml:/app/GenoPheno/config.yaml local/genopheno:0.95
(assuming server_example.zip is unpacked in the current directory)
