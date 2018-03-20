import tensorflow as tf

class pg_config():

    record           = False 

    # output config
    output_path  = "PG_results/"
    model_output = output_path + "model.weights/"
    #log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1
    
    
    # model and training config
    num_batches = 10 # number of batches trained on 
    batch_size = 50 # number of steps used to compute each policy update
    learning_rate = 3e-2
    #use_baseline = False 
    normalize_advantage=True 
    activation=tf.nn.relu
