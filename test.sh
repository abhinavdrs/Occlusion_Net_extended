docker run -v $PWD:/code --shm-size=32GB --runtime=nvidia $1 python infer.py -vid $2 -v True -cfg data/occlusion_net_test.yaml
