'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    # torchvision loads images as (C, H, W) but face_recognition expects (H, W, C)
    arr = prep_image_for_api(img)

    img_h, img_w = arr.shape[:2]
    total_pixels = float(img_h * img_w)

    # HOG model with upsample=2 catches smaller and farther away faces
    raw_locs = face_recognition.face_locations(arr, number_of_times_to_upsample=2, model="hog")

    if len(raw_locs) == 0:
        return detection_results

    # face_recognition returns (top, right, bottom, left) - put into tensor for batch processing
    loc_t = torch.tensor(raw_locs, dtype=torch.float32)

    t = loc_t[:, 0]
    r = loc_t[:, 1]
    b = loc_t[:, 2]
    l = loc_t[:, 3]

    bx = l
    by = t
    bw = r - l
    bh = b - t

    # two-stage filtering to remove false positives without hurting real detections:
    # 1) relative size - box must be at least 0.1% of total image area
    # 2) aspect ratio - real faces fall roughly between 0.4 and 2.5 width/height
    # Note: we avoid hard pixel minimums because some GT faces are genuinely small (18x26px)
    rel_size_ok = (bw * bh) >= (total_pixels * 0.001)
    ratio = bw / (bh + 1e-6)
    shape_ok = (ratio >= 0.4) & (ratio <= 2.5)

    valid = rel_size_ok & shape_ok

    all_boxes = torch.stack([bx, by, bw, bh], dim=1)

    for idx in range(all_boxes.shape[0]):
        if not valid[idx].item():
            continue
        b_vals = all_boxes[idx]
        detection_results.append([
            float(b_vals[0].item()),
            float(b_vals[1].item()),
            float(b_vals[2].item()),
            float(b_vals[3].item())
        ])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####

    torch.manual_seed(0)

    filenames = list(imgs.keys())
    face_vecs = []

    for fname in filenames:
        tensor = imgs[fname]

        # get numpy array in the right format for the API
        ready = prep_image_for_api(tensor)

        # detect where the face is first, then encode just that region
        # gives cleaner 128-d vectors than blind full-image encoding
        found_boxes = detect_faces(tensor)
        main_face = pick_biggest_face(found_boxes)

        enc_result = []

        if main_face is not None:
            fx, fy, fw, fh = main_face
            # convert [x,y,w,h] -> (top, right, bottom, left) for the API
            api_loc = [(int(fy), int(fx + fw), int(fy + fh), int(fx))]
            enc_result = face_recognition.face_encodings(
                ready,
                known_face_locations=api_loc,
                num_jitters=1,
                model="small"
            )

        # fallback: let face_recognition detect and encode on its own
        if len(enc_result) == 0:
            enc_result = face_recognition.face_encodings(
                ready,
                num_jitters=1,
                model="small"
            )

        # last resort: use the whole image as the face bounding box
        if len(enc_result) == 0:
            h_px, w_px = ready.shape[:2]
            whole_img = [(0, w_px, h_px, 0)]
            enc_result = face_recognition.face_encodings(
                ready,
                known_face_locations=whole_img,
                num_jitters=1,
                model="small"
            )

        if len(enc_result) == 0:
            vec = torch.zeros(128, dtype=torch.float32)
        else:
            vec = torch.tensor(enc_result[0], dtype=torch.float32)

        face_vecs.append(vec)

    # build (N x 128) feature matrix
    feat_mat = torch.stack(face_vecs, dim=0)

    # L2 normalize each row so dot product = cosine similarity
    # works much better for face embeddings than raw euclidean distance
    row_norms = feat_mat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    feat_mat = feat_mat / row_norms

    # run k-means from scratch with multiple restarts, keep lowest inertia
    winner_labels = None
    winner_score = float("inf")

    N_RESTARTS = 10
    MAX_STEPS = 200

    for run in range(N_RESTARTS):
        centers = init_centers_kmeanspp(feat_mat, K)
        cur_labels = torch.full((feat_mat.shape[0],), -1, dtype=torch.long)

        for step in range(MAX_STEPS):
            dist_mat = torch.cdist(feat_mat, centers, p=2)
            new_labels = torch.argmin(dist_mat, dim=1)

            if torch.equal(new_labels, cur_labels):
                break

            cur_labels = new_labels
            centers = recompute_centers(feat_mat, cur_labels, K, centers)

        cur_labels, centers = fix_empty_clusters(feat_mat, cur_labels, centers, K)

        assigned_centers = centers[cur_labels]
        score = ((feat_mat - assigned_centers) ** 2).sum().item()

        if score < winner_score:
            winner_score = score
            winner_labels = cur_labels.clone()

    for i, fname in enumerate(filenames):
        cluster_results[int(winner_labels[i].item())].append(fname)

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

def prep_image_for_api(img: torch.Tensor):
    # torchvision read_image gives C x H x W, face_recognition needs H x W x C uint8
    # clone().detach() avoids negative stride issues from torch.flip in bgr_to_rgb
    t = img.clone().detach()
    if t.dim() == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0).contiguous()
    else:
        t = t.contiguous()
    return t.to(torch.uint8).numpy().copy()


def pick_biggest_face(boxes: List[List[float]]):
    # returns the largest box by area - since clustering images have one face each,
    # this just grabs that face (and ignores background noise detections)
    if not boxes:
        return None
    biggest = boxes[0]
    best_area = biggest[2] * biggest[3]
    for box in boxes[1:]:
        a = box[2] * box[3]
        if a > best_area:
            best_area = a
            biggest = box
    return biggest


def init_centers_kmeanspp(data: torch.Tensor, k: int) -> torch.Tensor:
    # k-means++ init: pick first center randomly, then choose next centers
    # with probability proportional to squared distance from existing centers
    # this spreads them out and avoids bad starting configurations
    n = data.shape[0]
    seed_idx = torch.randint(0, n, (1,)).item()
    chosen = [data[seed_idx]]

    for _ in range(1, k):
        stacked = torch.stack(chosen, dim=0)
        all_dists = torch.cdist(data, stacked, p=2)
        nearest = all_dists.min(dim=1).values
        weights = (nearest ** 2).clamp(min=1e-12)
        weights = weights / weights.sum()
        pick = torch.multinomial(weights, 1).item()
        chosen.append(data[pick])

    return torch.stack(chosen, dim=0)


def recompute_centers(data: torch.Tensor, labels: torch.Tensor,
                      k: int, prev_centers: torch.Tensor) -> torch.Tensor:
    # update each centroid as the mean of its assigned points
    # re-normalize to unit sphere since we normalized the embeddings
    dim = data.shape[1]
    updated = torch.zeros((k, dim), dtype=data.dtype)

    for c in range(k):
        members = (labels == c)
        if members.sum().item() > 0:
            mean_vec = data[members].mean(dim=0)
            updated[c] = mean_vec / mean_vec.norm().clamp(min=1e-8)
        else:
            updated[c] = prev_centers[c]

    return updated


def fix_empty_clusters(data: torch.Tensor, labels: torch.Tensor,
                       centers: torch.Tensor, k: int):
    # if a cluster lost all its members, steal the farthest point from
    # whichever cluster it currently belongs to and reassign it
    for c in range(k):
        if (labels == c).sum().item() == 0:
            dist_mat = torch.cdist(data, centers, p=2)
            assigned_dists = dist_mat[torch.arange(data.shape[0]), labels.clamp(min=0)]
            worst = torch.argmax(assigned_dists).item()
            labels[worst] = c
            centers = recompute_centers(data, labels, k, centers)

    return labels, centers