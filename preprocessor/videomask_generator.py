import torch
import cv2
import numpy as np
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from PIL import Image, ImageOps
import random
import os
import requests
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from typing import Union, Optional, Tuple, List

# --- Helper Functions (Unchanged) ---
def tensor_to_cv2_img(tensor_frame: torch.Tensor) -> np.ndarray:
    """Converts a single RGB torch tensor to a BGR OpenCV image."""
    img_np = (tensor_frame.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def tensor_to_cv2_bgra_img(tensor_frame: torch.Tensor) -> np.ndarray:
    """Converts a single RGBA torch tensor to a BGRA OpenCV image."""
    if tensor_frame.shape[2] != 4:
        raise ValueError("Input tensor must be an RGBA image with 4 channels.")
    img_np = (tensor_frame.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL image to a torch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)

class VideoMaskGenerator:
    def __init__(self, antelopv2_path=".", device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        providers = ["CUDAExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
        
        # Initialize face detection and landmark model (antelopev2 provides both)
        self.detection_model = FaceAnalysis(name="antelopev2", root=antelopv2_path, providers=providers)
        self.detection_model.prepare(ctx_id=0, det_size=(640, 640))

        # Initialize face parsing model
        self.parsing_model = init_parsing_model(model_name="bisenet", device=self.device)
        self.parsing_model.eval()
        
        print("FaceProcessor initialized successfully.")

    def process(
        self,
        video_path: str,
        face_image: Union[str, Image.Image],
        confidence_threshold: float = 0.5,
        face_crop_scale: float = 1.5,
        dilation_kernel_size: int = 10,
        feather_amount: int = 21,
        random_horizontal_flip_chance: float = 0.0,
        match_angle_and_size: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        """
        Processes a video to replace a face with a provided face image.

        Args:
            video_path (str): Path to the input video file.
            face_image (Union[str, Image.Image]): Path or PIL image of the face to paste.
            confidence_threshold (float): Confidence threshold for face detection.
            face_crop_scale (float): Scale factor for cropping the detected face box.
            dilation_kernel_size (int): Kernel size for mask dilation.
            feather_amount (int): Amount of feathering for the mask edges.
            random_horizontal_flip_chance (float): Chance to flip the source face horizontally.
            match_angle_and_size (bool): Whether to use landmark matching for rotation and scale.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, int, int]: 
                - Processed video as a numpy array (F, H, W, C).
                - Generated masks as a numpy array (F, H, W).
                - Width of the processed video.
                - Height of the processed video.
                - Number of frames in the processed video.
        """
        # --- Video Pre-processing ---
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames:
            raise ValueError("Could not read any frames from the video.")

        video_np = np.array(frames)
        
        h, w = video_np.shape[1], video_np.shape[2]
        new_h, new_w = (h // 16) * 16, (w // 16) * 16
        
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        video_cropped = video_np[:, y_start:y_start+new_h, x_start:x_start+new_w, :]

        num_frames = video_cropped.shape[0]
        target_frames = (num_frames // 4) * 4 + 1
        video_trimmed = video_cropped[:target_frames]

        final_h, final_w, final_frames = video_trimmed.shape[1], video_trimmed.shape[2], video_trimmed.shape[0]
        print(f"Video pre-processed: {final_w}x{final_h}, {final_frames} frames.")

        # --- Face Image Pre-processing & Source Landmark Extraction ---
        if isinstance(face_image, str):
            if face_image.startswith("http"):
                face_image = Image.open(requests.get(face_image, stream=True, timeout=10).raw)
            else:
                face_image = Image.open(face_image)
        
        face_image = ImageOps.exif_transpose(face_image).convert("RGBA")
        face_rgba_tensor = pil_to_tensor(face_image)
        face_to_paste_cv2 = tensor_to_cv2_bgra_img(face_rgba_tensor)

        source_kpts = None
        if match_angle_and_size:
            # Use insightface (antelopev2) to get landmarks from the source face image
            source_face_bgr = cv2.cvtColor(face_to_paste_cv2, cv2.COLOR_BGRA2BGR)
            source_faces = self.detection_model.get(source_face_bgr)
            if source_faces:
                # Use the landmarks from the first (and likely only) detected face
                source_kpts = source_faces[0].kps
            else:
                print("[Warning] No face or landmarks found in source image. Disabling angle matching.")
                match_angle_and_size = False
        
        face_to_paste_pil = Image.fromarray((face_rgba_tensor.cpu().numpy() * 255).astype(np.uint8), 'RGBA')

        # --- Main Processing Loop ---
        processed_frames_list = []
        mask_list = []

        for i in tqdm(range(final_frames), desc="Pasting face onto frames"):
            frame_rgb = video_trimmed[i]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Use insightface for detection and landmarks
            faces = self.detection_model.get(frame_bgr)
            
            pasted = False
            final_mask = np.zeros((final_h, final_w), dtype=np.uint8)

            if faces:
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                
                if largest_face.det_score > confidence_threshold:
                    # **MODIFIED BLOCK**: Use insightface landmarks for affine transform
                    if match_angle_and_size and source_kpts is not None:
                        target_kpts = largest_face.kps # Get landmarks directly from the detected face
                        
                        # Estimate the transformation matrix
                        M, _ = cv2.estimateAffinePartial2D(source_kpts, target_kpts, method=cv2.LMEDS)
                        
                        if M is not None:
                            # Split the RGBA source face for separate warping
                            b, g, r, a = cv2.split(face_to_paste_cv2)
                            source_rgb_cv2 = cv2.merge([r, g, b])
                            
                            # Warp the face and its alpha channel
                            warped_face = cv2.warpAffine(source_rgb_cv2, M, (final_w, final_h))
                            warped_alpha = cv2.warpAffine(a, M, (final_w, final_h))
                            
                            # Blend the warped face onto the frame using the warped alpha channel
                            alpha_float = warped_alpha.astype(np.float32) / 255.0
                            alpha_expanded = np.expand_dims(alpha_float, axis=2)
                            
                            frame_rgb = (1.0 - alpha_expanded) * frame_rgb + alpha_expanded * warped_face
                            frame_rgb = frame_rgb.astype(np.uint8)
                            final_mask = warped_alpha
                            pasted = True

                    # Fallback to simple box-pasting if angle matching is off or fails
                    if not pasted: 
                        x1, y1, x2, y2 = map(int, largest_face.bbox)
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        side_len = int(max(x2 - x1, y2 - y1) * face_crop_scale)
                        half_side = side_len // 2
                        
                        crop_y1, crop_x1 = max(center_y - half_side, 0), max(center_x - half_side, 0)
                        crop_y2, crop_x2 = min(center_y + half_side, final_h), min(center_x + half_side, final_w)
                        
                        box_w, box_h = crop_x2 - crop_x1, crop_y2 - crop_y1

                        if box_w > 0 and box_h > 0:
                            source_img = face_to_paste_pil.copy()
                            if random.random() < random_horizontal_flip_chance:
                                source_img = source_img.transpose(Image.FLIP_LEFT_RIGHT)
                            
                            face_resized = source_img.resize((box_w, box_h), Image.Resampling.LANCZOS)
                            
                            target_frame_pil = Image.fromarray(frame_rgb)
                            
                            # --- Mask Generation using BiSeNet ---
                            face_crop_bgr = cv2.cvtColor(frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_RGB2BGR)
                            if face_crop_bgr.size > 0:
                                face_resized_512 = cv2.resize(face_crop_bgr, (512, 512), interpolation=cv2.INTER_AREA)
                                face_rgb_512 = cv2.cvtColor(face_resized_512, cv2.COLOR_BGR2RGB)
                                face_tensor_in = torch.from_numpy(face_rgb_512.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
                                
                                with torch.no_grad():
                                    normalized_face = normalize(face_tensor_in, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    parsing_map = self.parsing_model(normalized_face)[0].argmax(dim=1, keepdim=True)
                                
                                parsing_map_np = parsing_map.squeeze().cpu().numpy().astype(np.uint8)
                                parts_to_include = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # All face parts
                                final_mask_512 = np.isin(parsing_map_np, parts_to_include).astype(np.uint8) * 255
                                
                                if dilation_kernel_size > 0:
                                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
                                    final_mask_512 = cv2.dilate(final_mask_512, kernel, iterations=1)
                                
                                if feather_amount > 0:
                                    if feather_amount % 2 == 0: feather_amount += 1
                                    final_mask_512 = cv2.GaussianBlur(final_mask_512, (feather_amount, feather_amount), 0)
                                
                                mask_resized_to_crop = cv2.resize(final_mask_512, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
                                generated_mask_pil = Image.fromarray(mask_resized_to_crop, mode='L')
                                
                                target_frame_pil.paste(face_resized, (crop_x1, crop_y1), mask=generated_mask_pil)
                                frame_rgb = np.array(target_frame_pil)
                                final_mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask_resized_to_crop

            processed_frames_list.append(frame_rgb)
            mask_list.append(final_mask)

        output_video = np.stack(processed_frames_list)
        # Ensure mask has a channel dimension for consistency
        output_masks = np.stack(mask_list)[..., np.newaxis] 
        
        return (output_video, output_masks, final_w, final_h, final_frames)