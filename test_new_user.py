import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from model import Heater
import pickle

def test_new_user(user_features, model_path='./checkpoints/lastfm_model/model.ckpt-28'):
    # Load model parameters
    with open('./data/LastFM/info.pkl', 'rb') as f:
        info = pickle.load(f)
        num_user = info['num_user']
        num_item = info['num_item']
    
    # Load item preferences
    v_pref = np.load('./data/LastFM/V_BPR.npy')
    
    # Create sparse feature vector for the new user
    user_content = sp.csr_matrix(user_features)
    
    # Model parameters (should match your training parameters)
    model_params = {
        'latent_rank_in': v_pref.shape[1],  # Same as BPR embedding dimension
        'user_content_rank': user_content.shape[1],
        'item_content_rank': 0,  # LastFM doesn't use item content
        'model_select': [200],  # Hidden layer dimensions
        'rank_out': 200,  # Output embedding dimension
        'reg': 0.001,  # Regularization parameter
        'alpha': 0.0001,  # Learning rate
        'dim': 5  # Number of experts
    }
    
    # Create model
    model = Heater(**model_params)
    model.build_model()
    model.build_predictor(recall_at=[20, 50, 100])
    
    # Create TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    
    # Prepare feed dict for the new user
    feed_dict = {
        model.Ucontent: user_content.toarray(),
        model.dropout_user_indicator: np.zeros((1, 1)),  # New user, so use content
        model.Vin: v_pref,
        model.is_training: False,
        model.Uin: np.zeros((1, v_pref.shape[1]))  # Add zero embeddings for new user
    }
    
    # Get recommendations
    preds = sess.run(model.eval_preds_cold, feed_dict=feed_dict)
    
    return preds[0]  # Return top-k recommendations for the user

# Example usage:
if __name__ == "__main__":
    # Create example user features (modify based on your actual feature format)
    # In LastFM, user features are based on social connections
    num_users = 1892  # From the user_content shape
    new_user_features = np.zeros(num_users)
    # Set some random connections (in practice, these would be real social connections)
    random_friends = np.random.choice(num_users, 20, replace=False)
    new_user_features[random_friends] = 1
    
    print("\nNew user's social connections:")
    print(f"Connected to {len(random_friends)} users: {random_friends}")
    
    # Load user listening history to understand friends' music preferences
    user_artists = {}
    with open('./data/LastFM/user_artists.dat', 'r') as f:
        next(f)  # Skip header
        for line in f:
            user_id, artist_id, weight = map(int, line.strip().split('\t'))
            if user_id not in user_artists:
                user_artists[user_id] = []
            user_artists[user_id].append((artist_id, weight))
    
    print("\nTop artists for each friend:")
    for friend_id in random_friends:
        if friend_id in user_artists:
            # Get top 3 most listened artists for this friend
            top_artists = sorted(user_artists[friend_id], key=lambda x: x[1], reverse=True)[:3]
            print(f"\nFriend {friend_id}:")
            for artist_id, weight in top_artists:
                print(f"  Artist {artist_id} (listened {weight} times)")
    
    # Get recommendations
    top_k_items = test_new_user(new_user_features)
    
    print("\nTop 10 recommended artist IDs for new user:")
    for i, item_id in enumerate(top_k_items[:10], 1):
        print(f"{i}. Artist ID: {item_id}")
        
    print("\nNote: You can look up these artist IDs in the LastFM dataset to find their names.")
