import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def _resize_volume(volume, target_shape):
    """3D 볼륨을 목표 크기로 리사이즈"""
    if volume.shape == target_shape:
        return volume
        
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom_factors, order=1, mode='nearest')


class GradCAM:
    """3D Grad-CAM 생성 핵심 로직"""
    
    def __init__(self, model, target_layer_name=None):
        """GradCAM 초기화 및 Hook 등록"""
        self.model = model
        self.model.train()  # 그래디언트 계산용
        
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self.target_layer_name = target_layer_name if target_layer_name else self._find_target_conv_layer()
        self._register_hooks()

    def _find_target_conv_layer(self):
        """Grad-CAM용 마지막 Conv3d 레이어 탐색"""
        model_to_search = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 특정 모델 구조 우선 확인
        if hasattr(model_to_search, 'backbone'):
            if 'nnUNet' in type(model_to_search.backbone).__name__:
                return 'backbone.encoder_module.stages.-1'
            if hasattr(model_to_search.backbone, 'layer4'):
                return 'backbone.layer4'

        # 일반적인 경우: 마지막 Conv3d 레이어 선택
        last_conv_name = None
        for name, module in model_to_search.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                last_conv_name = name
        
        if last_conv_name:
            return last_conv_name
        raise ValueError("모델에서 Conv3d 레이어를 찾을 수 없습니다.")

    def _get_layer_from_name(self, layer_name):
        """문자열 레이어 경로로 실제 레이어 객체 반환"""
        model_to_search = self.model.module if hasattr(self.model, 'module') else self.model
        current_layer = model_to_search
        for name in layer_name.split('.'):
            if name.isdigit() or (name.startswith('-') and name[1:].isdigit()):
                current_layer = current_layer[int(name)]
            else:
                current_layer = getattr(current_layer, name)
        return current_layer

    def _register_hooks(self):
        """대상 레이어에 Forward/Backward Hook 부착"""
        target_layer = self._get_layer_from_name(self.target_layer_name)

        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor, target_class=None):
        """입력 텐서에 대한 Class Activation Map 생성"""
        input_tensor.requires_grad_(True)
        output = self.model(images=input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hook을 통해 그래디언트나 활성맵이 캡처되지 않았습니다.")
        
        # 그래디언트로 채널별 가중치 계산 (Global Average Pooling)
        weights = np.mean(self.gradients.cpu().numpy()[0], axis=(1, 2, 3), keepdims=True)
        
        # 가중치와 피처맵을 곱하여 CAM 계산
        activations_np = self.activations.cpu().numpy()[0]
        cam = np.sum(weights * activations_np, axis=0)
        
        # ReLU 적용 및 정규화
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, target_class
    
    def cleanup(self):
        """Hook 제거 및 모델 모드 복원"""
        for handle in self.hook_handles:
            handle.remove()
        self.model.eval()


class CAMVisualizer:
    """3D CAM 시각화 결과물 생성"""
    
    def __init__(self, save_dir, alpha=0.4, num_key_slices=12):
        """시각화 클래스 초기화"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.alpha = alpha
        self.num_key_slices = num_key_slices

    def run_visualizations(self, image, cam, filename_prefix, target_class, class_names, true_class=None):
        """모든 시각화 실행 및 저장"""
        self._save_key_slices_grid(image, cam, filename_prefix, target_class, class_names, true_class)
        self._save_all_slices_grid(image, cam, filename_prefix, target_class, class_names, true_class)
        self._save_3d_projection(image, cam, filename_prefix, target_class, class_names, true_class)

    def _create_slice_overlay(self, image_slice, cam_slice):
        """단일 2D 슬라이스에 이미지와 CAM 오버레이"""
        image_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-6)
        cam_colored = plt.cm.jet(cam_slice)[:, :, :3]
        overlay = (1 - self.alpha) * np.stack([image_norm] * 3, axis=-1) + self.alpha * cam_colored
        return np.clip(overlay, 0, 1)

    def _save_plot(self, fig, title, save_path):
        """Matplotlib Figure 저장"""
        fig.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def _save_key_slices_grid(self, image, cam, filename_prefix, target_class, class_names, true_class=None):
        """CAM 활성화가 높은 주요 슬라이스 격자 이미지 저장"""
        depth = image.shape[0]
        
        # 주요 슬라이스 선택
        if depth <= self.num_key_slices:
            key_indices = list(range(depth))
        else:
            cam_scores = np.mean(cam, axis=(1, 2))
            top_indices = np.argsort(cam_scores)[-self.num_key_slices:]
            key_indices = sorted(list(top_indices))

        if not key_indices:
            return

        # 격자 레이아웃 계산
        num_slices = len(key_indices)
        grid_cols = 4
        grid_rows = int(np.ceil(num_slices / grid_cols))
        
        fig, axes = plt.subplots(grid_rows, grid_cols * 3, figsize=(grid_cols * 5, grid_rows * 5))
        axes = axes.flatten() if axes.ndim > 1 else [axes]

        for i in range(grid_rows * grid_cols):
            base_idx = i * 3
            if i < num_slices and base_idx + 2 < len(axes):
                slice_idx = key_indices[i]
                
                # 원본, CAM, 오버레이 이미지 표시
                axes[base_idx].imshow(image[slice_idx], cmap='gray')
                axes[base_idx].set_title(f'Original {slice_idx}', fontsize=8)
                axes[base_idx+1].imshow(cam[slice_idx], cmap='jet', vmin=0, vmax=1)
                axes[base_idx+1].set_title(f'CAM {slice_idx}', fontsize=8)
                overlay = self._create_slice_overlay(image[slice_idx], cam[slice_idx])
                axes[base_idx+2].imshow(overlay)
                axes[base_idx+2].set_title(f'Overlay {slice_idx}', fontsize=8)
            
            # 축 비활성화
            for j in range(3):
                if base_idx + j < len(axes):
                    axes[base_idx + j].axis('off')
        
        title_parts = [f'Key Slices (Top {len(key_indices)})']
        if true_class is not None:
            title_parts.append(f'True: {class_names[true_class]}')
        title_parts.append(f'Pred: {class_names[target_class]}')
        title = ' - '.join(title_parts)
        save_path = os.path.join(self.save_dir, f'{filename_prefix}_key_slices.png')
        self._save_plot(fig, title, save_path)

    def _save_all_slices_grid(self, image, cam, filename_prefix, target_class, class_names, true_class=None):
        """모든 슬라이스에 대한 격자 이미지를 저장"""
        depth = image.shape[0]
        slice_indices = list(range(depth))
        grid_cols = int(np.ceil(np.sqrt(depth)))
        
        if not slice_indices:
            return

        # 격자 레이아웃 계산
        num_slices = len(slice_indices)
        grid_rows = int(np.ceil(num_slices / grid_cols))
        
        fig, axes = plt.subplots(grid_rows, grid_cols * 3, figsize=(grid_cols * 5, grid_rows * 5))
        axes = axes.flatten() if axes.ndim > 1 else [axes]

        for i in range(grid_rows * grid_cols):
            base_idx = i * 3
            if i < num_slices and base_idx + 2 < len(axes):
                slice_idx = slice_indices[i]
                
                # 원본, CAM, 오버레이 이미지 표시
                axes[base_idx].imshow(image[slice_idx], cmap='gray')
                axes[base_idx].set_title(f'Original {slice_idx}', fontsize=8)
                axes[base_idx+1].imshow(cam[slice_idx], cmap='jet', vmin=0, vmax=1)
                axes[base_idx+1].set_title(f'CAM {slice_idx}', fontsize=8)
                overlay = self._create_slice_overlay(image[slice_idx], cam[slice_idx])
                axes[base_idx+2].imshow(overlay)
                axes[base_idx+2].set_title(f'Overlay {slice_idx}', fontsize=8)
            
            # 축 비활성화
            for j in range(3):
                if base_idx + j < len(axes):
                    axes[base_idx + j].axis('off')
        
        title_parts = ['All Slices']
        if true_class is not None:
            title_parts.append(f'True: {class_names[true_class]}')
        title_parts.append(f'Pred: {class_names[target_class]}')
        title = ' - '.join(title_parts)
        save_path = os.path.join(self.save_dir, f'{filename_prefix}_all_slices.png')
        self._save_plot(fig, title, save_path)

    def _save_3d_projection(self, image, cam, filename_prefix, target_class, class_names, true_class=None):
        """3D 볼륨의 2D 투영(MIP) 이미지 저장"""
        views = {'Axial': 0, 'Coronal': 1, 'Sagittal': 2}
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (view_name, axis) in enumerate(views.items()):
            # Maximum Intensity Projection
            img_mip = np.max(image, axis=axis)
            cam_mip = np.max(cam, axis=axis)

            # 해부학적 방향 조정
            if view_name in ['Coronal', 'Sagittal']:
                img_mip = np.rot90(img_mip, 2)
                cam_mip = np.rot90(cam_mip, 2)
                # 세로 리사이즈로 가시성 향상
                if img_mip.shape[0] < 64:
                    resize_factor = 64 / img_mip.shape[0]
                    img_mip = zoom(img_mip, (resize_factor, 1), order=1) 
                    cam_mip = zoom(cam_mip, (resize_factor, 1), order=1)
            elif view_name == 'Axial':
                img_mip = np.fliplr(img_mip)
                cam_mip = np.fliplr(cam_mip)

            overlay = self._create_slice_overlay(img_mip, cam_mip)
            
            axes[0, i].imshow(img_mip, cmap='gray')
            axes[0, i].set_title(f'Original {view_name} MIP')
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'Overlay {view_name} MIP')
        
        for ax in axes.flatten(): 
            ax.axis('off')

        title_parts = ['3D Maximum Intensity Projection']
        if true_class is not None:
            title_parts.append(f'True: {class_names[true_class]}')
        title_parts.append(f'Pred: {class_names[target_class]}')
        title = ' - '.join(title_parts)
        save_path = os.path.join(self.save_dir, f'{filename_prefix}_3d_projection.png')
        self._save_plot(fig, title, save_path)


def generate_cam_for_sample(model, image_tensor, target_class, class_names, save_dir, sample_name, true_label=None):
    """단일 샘플에 대한 Grad-CAM 생성 및 시각화"""
    original_model_mode = model.training
    gradcam = GradCAM(model)
    
    try:
        # Grad-CAM 생성
        cam_input = image_tensor.unsqueeze(0) if image_tensor.dim() == 4 else image_tensor
        cam, predicted_class = gradcam.generate_cam(cam_input, target_class)
        if cam is None: 
            return None, None

        # CAM을 원본 이미지 크기로 리사이즈
        input_shape_3d = image_tensor.shape[1:]
        cam_resized = _resize_volume(cam, target_shape=input_shape_3d)
        
        # 시각화 실행
        image_for_viz = image_tensor.cpu().numpy()[0]
        visualizer = CAMVisualizer(save_dir)
        visualizer.run_visualizations(
            image=image_for_viz,
            cam=cam_resized,
            filename_prefix=sample_name,
            target_class=predicted_class,
            class_names=class_names,
            true_class=true_label
        )
        return cam_resized, predicted_class

    except Exception as e:
        print(f"{sample_name} CAM 생성 오류: {e}")
        return None, None
    
    finally:
        # Hook 제거 및 모델 모드 복원
        gradcam.cleanup()
        model.train(original_model_mode)