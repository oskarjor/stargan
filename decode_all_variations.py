import os
import numpy as np
import json

def file_to_repr(file_name, num_attributes):
    representation = file_name.split("_")
    z_vector = representation[1:num_attributes+1]
    remainder = representation[num_attributes+1:]
    prob = remainder[1]
    target = remainder[3]
    realness = remainder[5].split(".png")[0]
    return z_vector, prob, target, realness

def decode_all_variations(folder_path):
    config = json.load(open(os.path.join(folder_path, "config.json")))
    num_attributes = len(config["attributes"])
    # Get all subfolders in the given path
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    # Sort subfolders numerically
    subfolders.sort(key=lambda x: int(os.path.basename(x)))
    all_z_vectors = []
    all_probs = []
    all_targets = []
    all_realnesses = []
    for subfolder in subfolders:
        single_image_z_vectors = []
        single_image_probs = []
        single_image_targets = []
        single_image_realnesses = []
        for file in os.scandir(subfolder):
            if file.is_file() and file.name.endswith(".png") and file.name.startswith("z_"):
                z_vector, prob, target, realness = file_to_repr(file.name, num_attributes)
                single_image_z_vectors.append(z_vector)
                single_image_probs.append(prob)
                single_image_targets.append(target)
                single_image_realnesses.append(realness)
        all_z_vectors.append(single_image_z_vectors)
        all_probs.append(single_image_probs)
        all_targets.append(single_image_targets)
        all_realnesses.append(single_image_realnesses)
    npy_z_vectors = np.array(all_z_vectors)
    npy_probs = np.array(all_probs)
    npy_targets = np.array(all_targets)
    npy_realnesses = np.array(all_realnesses)
    return npy_z_vectors, npy_probs, npy_targets, npy_realnesses

def score_all_variations(mean_losses, mean_realnesses):
    min_loss = np.min(mean_losses)
    min_loss_idx = np.argmin(mean_losses)
    max_realness = np.max(mean_realnesses)
    max_realness_idx = np.argmax(mean_realnesses)
    return min_loss, min_loss_idx, max_realness, max_realness_idx
    

def get_stats(npy_loss, npy_realness):
    return {
        "mean_loss": np.mean(npy_loss),
        "mean_realness": np.mean(npy_realness),
        "std_loss": np.std(npy_loss),
        "std_realness": np.std(npy_realness),
    }

def main(folder_path, p_accuracy, p_realness):
    npy_z_vectors, npy_probs, npy_targets, npy_realnesses = decode_all_variations(folder_path)
    config = json.load(open(os.path.join(folder_path, "config.json")))
    print(config)
    # if config["classification_attr"] in config["attributes"]:
    #     print(f"Classification attribute {config['classification_attr']} found in attributes, updating targets...")
    #     if np.allclose(npy_targets, npy_z_vectors[:, :, config["attributes"].index(config["classification_attr"])]):
    #         print("Targets and z vectors are the same, no need to update")
    #     else:
    #         print("Targets and z vectors are different, updating targets...")
    #         npy_targets = npy_z_vectors[:, :, config["attributes"].index(config["classification_attr"])]
    # Convert probabilities and targets to float for comparison
    npy_z_vectors = npy_z_vectors.astype(int)
    npy_probs = npy_probs.astype(float)
    npy_targets = npy_targets.astype(int)
    npy_realnesses = npy_realnesses.astype(float)


    # Round probabilities to get predictions (threshold at 0.5)
    predictions = (npy_probs >= 0.5).astype(int)
    
    # Calculate accuracy for each image
    correct_predictions = (predictions == npy_targets)
    accuracy_per_image = np.mean(correct_predictions, axis=1)
    
    # Get indices of images with accuracy above 0.5 (default threshold)
    high_accuracy_indices = np.where(accuracy_per_image > p_accuracy)[0]
    
    print(f"\nImages with accuracy > {p_accuracy}:")
    print(f"Found {len(high_accuracy_indices)} images")
    print(f"Image indices: {high_accuracy_indices}")
    
    # Calculate mean accuracy across all high performing images
    mean_accuracy = np.mean(accuracy_per_image[high_accuracy_indices])
    print(f"Mean accuracy for these images: {mean_accuracy:.3f}")

    per_image_realness = np.mean(npy_realnesses, axis=1)
    # Get indices of images with high realness scores
    realness_threshold = np.percentile(per_image_realness, p_realness*100)
    high_realness_indices = np.where(per_image_realness > realness_threshold)[0]

    print(f"\nImages with realness in top {p_realness*100:.1f}%:")
    print(f"Found {len(high_realness_indices)} images")
    print(f"Image indices: {high_realness_indices}")
    
    # Calculate mean realness across high performing images
    mean_realness = np.mean(per_image_realness[high_realness_indices])
    print(f"Mean realness for these images: {mean_realness:.3f}")

    # Find intersection of high accuracy and high realness indices
    good_images = np.intersect1d(high_accuracy_indices, high_realness_indices)
    
    print(f"\nImages that have both high accuracy and high realness:")
    print(f"Found {len(good_images)} images")
    print(f"Image indices: {good_images}")
    
    # Calculate mean metrics for these images
    mean_accuracy_good = np.mean(accuracy_per_image[good_images])
    mean_realness_good = np.mean(per_image_realness[good_images])
    print(f"Mean accuracy for these images: {mean_accuracy_good:.3f}")
    print(f"Mean realness for these images: {mean_realness_good:.3f}")

    # find high realness images that do not have high accuracy
    bad_images = np.setdiff1d(high_realness_indices, high_accuracy_indices)
    print(f"\nImages that have high realness but low accuracy:")
    print(f"Found {len(bad_images)} images")
    print(f"Image indices: {bad_images}")
    
    # Calculate mean metrics for these images
    mean_accuracy_bad = np.mean(accuracy_per_image[bad_images])
    mean_realness_bad = np.mean(per_image_realness[bad_images])
    print(f"Mean accuracy for these images: {mean_accuracy_bad:.3f}")
    print(f"Mean realness for these images: {mean_realness_bad:.3f}")


if __name__ == "__main__":
    # folder_path = "generated_images/bal_gray/young_noise_00/model_10_only_relevant_z_vectors"
    p_accuracy = 0.5
    p_realness = 0.95
    folder_path = "cherry_picked"
    main(folder_path, p_accuracy, p_realness)
