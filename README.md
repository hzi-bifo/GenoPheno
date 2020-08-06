# GenoPheno

docker build --tag local/genopheno:0.95 .
docker run -v /root/git/GenoPheno:/root/git/GenoPheno -v /root/git/GenoPheno/out:/out local/genopheno:0.95 /root/git/GenoPheno/data/server_example.zip /out/hml_file /out
