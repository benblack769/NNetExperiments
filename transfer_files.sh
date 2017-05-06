scp -ri  ~/.ssh/theano_key.pem ubuntu@ec2-34-208-21-170.us-west-2.compute.amazonaws.com:"~/music_net/plots/plot_data/" ./load_files
scp -ri  ~/.ssh/theano_key.pem ubuntu@ec2-34-208-21-170.us-west-2.compute.amazonaws.com:"~/music_net/saved_weights/" ./load_files


#scp -ri ~/.ssh/theano_key.pem  ./saved_weights/ ubuntu@ec2-34-208-21-170.us-west-2.compute.amazonaws.com:"~/music_net/saved_weights/"
