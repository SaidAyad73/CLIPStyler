import streamlit as st
import os
import tempfile
from datetime import datetime
from utils_t import train
import numpy as np

st.set_page_config(page_title="CLIPStyler", page_icon="🎨", layout="wide")

st.title("🎨 CLIPStyler - Fast Style Training")
st.write("Train neural style transfer on your images using natural language prompts")

# Initialize session state
if 'done' not in st.session_state:
    st.session_state.done = False

# Left column: inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Inputs")
    
    # Mode selection
    mode = st.radio("Mode:", ["Train on Uploads", "Full Training (Directory)", "Predict (Inference)"], horizontal=False)
    
    uploaded_files = st.file_uploader(
        "Upload image(s) (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    ) if mode in ["Train on Uploads", "Predict (Inference)"] else None
    
    data_dir = st.text_input(
        "Root directory path (all images will be used)",
        placeholder="/path/to/images or C:\\path\\to\\images"
    ) if mode == "Full Training (Directory)" else None
    
    prompts_text = st.text_area(
        "Style prompts (one per line)",
        height=80,
        placeholder="oil painting\nimpressionist\nwatercolor"
    ) if mode != "Predict (Inference)" else None
    
    prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()] if prompts_text else []
    
    # Checkpoint upload
    checkpoint_file = st.file_uploader(
        "Load checkpoint (.ckpt) - Optional: continue from existing model",
        type=["ckpt", "pt"]
    ) if mode in ["Train on Uploads", "Full Training (Directory)"] else None
    
    # For prediction mode
    if mode == "Predict (Inference)":
        checkpoint_file = st.file_uploader(
            "Load checkpoint (.ckpt)",
            type=["ckpt", "pt"]
        )

with col2:
    st.subheader("⚙️ Parameters")
    
    col_a, col_b = st.columns(2)
    with col_a:
        n_epochs = st.slider("Epochs", 1, 50, 10)
        lambda_clip = st.slider(
            "λ CLIP (style strength)",
            10, 5000, 500,
            help="↑ Higher = stronger style transfer. Too high = artifacts. Try 500-2000"
        )
        lambda_content = st.slider(
            "λ Content (preserve original)",
            10, 500, 150,
            help="↑ Higher = keeps more of input image. Too high = no style applied"
        )
        lr_options = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        lr_labels = ["1e-5", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
        lr_idx = st.select_slider("Learning Rate", options=range(len(lr_options)), format_func=lambda i: lr_labels[i], value=2)
        lr = lr_options[lr_idx]
        scale_data = st.number_input(
            "Scale Data (dataset multiplication factor)",
            min_value=1,
            max_value=1000,
            value=1,
            step=1,
            help="Repeat dataset N times to amortize training overhead (1 = no repeat)"
        )
    
    with col_b:
        batch_size = st.slider("Batch Size", 1, 8, 1)
        lambda_patch = st.slider(
            "λ Patch (spatial smoothness)",
            100, 20000, 5000,
            help="↑ Higher = smoother local regions. Usually ~10× λ CLIP"
        )
        lambda_tv = st.slider(
            "λ TV (noise reduction)",
            1e-3, 1e-1, 2e-3, format="%.4f",
            help="↑ Higher = less noise/artifacts but less detail"
        )
        img_size = st.slider("Image Size", 256, 1024, 512, 64)
    
    with st.expander("Advanced"):
        patch_size = st.slider("Patch Size", 32, 256, 64, 16)
        n_patches = st.slider("Patches/Step", 8, 128, 64, 8)
        accumulate = st.slider("Grad Accumulate", 1, 16, 1)
        compile_model = st.checkbox("Compile Model", False)
        n_devices = st.slider("GPUs", 1, 4, 1)
        loader_workers = st.slider("DataLoader Workers", 0, 8, 0)
    
    with st.expander("ℹ️ Lambda Guide"):
        st.markdown("""
        **Understanding Loss Weights (lambdas):**
        
        - **λ CLIP**: How much the model follows your style prompt
          - Low (100-300): Subtle style, keeps original look
          - Medium (500-1000): Balanced style transfer
          - High (2000+): Strong style, may distort structure
        
        - **λ Content**: Preserves the original image structure
          - Low (50): Radical transformation
          - Medium (150): Balanced content+style
          - High (300+): Mostly keeps original, subtle style
        
        - **λ Patch**: Smoothness across patches
          - Usually set to ~10× λ CLIP for consistency
          - Higher = smoother but less detail
        
        - **λ TV**: Removes noise and speckles
          - Default (0.002): Light smoothing
          - Higher (0.01): Aggressive noise removal
          - Lower (0.0001): Keep all details
        
        **Quick Presets:**
        - **Subtle**: CLIP=300, Content=200
        - **Balanced**: CLIP=500, Content=150
        - **Strong**: CLIP=1500, Content=75
        """)

st.divider()

col_train, col_status = st.columns([1, 2])

with col_train:
    if mode == "Train on Uploads":
        button_label = "▶️ Train"
        can_train = uploaded_files and prompts
    elif mode == "Full Training (Directory)":
        button_label = "▶️ Train"
        can_train = data_dir and prompts and os.path.isdir(data_dir)
    else:  # Predict (Inference)
        button_label = "▶️ Predict"
        can_train = checkpoint_file and uploaded_files
    
    if st.button(button_label, key="train_btn", disabled=not can_train):
        if mode == "Train on Uploads":
            if not uploaded_files:
                st.error("Upload at least one image")
            elif not prompts:
                st.error("Enter at least one prompt")
            else:
                image_paths = []
                temp_dir = tempfile.gettempdir()
                for file in uploaded_files:
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    image_paths.append(temp_path)
                
                # Save checkpoint if provided
                checkpoint_path = None
                if checkpoint_file:
                    checkpoint_path = os.path.join(temp_dir, checkpoint_file.name)
                    with open(checkpoint_path, "wb") as f:
                        f.write(checkpoint_file.getbuffer())
                
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_short = "_".join([p[:10] for p in prompts[:2]])
                save_name = f"clipstyler_{prompt_short}_{ts}.ckpt"
                
                with st.spinner("Training... this may take a few minutes"):
                    try:
                        model = train(
                            paths=image_paths,
                            prompts=prompts,
                            patch_size=patch_size,
                            scale_data=scale_data,
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            n_patches=n_patches,
                            img_size=img_size,
                            lambda_clip=lambda_clip,
                            lambda_patch=lambda_patch,
                            lambda_content=lambda_content,
                            lambda_tv=lambda_tv,
                            lr=lr,
                            save_name=save_name,
                            accumulate_grad_batches=accumulate,
                            compile_model=compile_model,
                            use_logger=True,
                            loader_process=loader_workers,
                            n_devices=n_devices,
                            checkpoint_path=checkpoint_path,
                            restart_epochs=10,
                        )
                        st.session_state.done = True
                        st.success(f"✅ Done! Checkpoint: {save_name}")
                        print(f'✅ Training complete! Checkpoint saved as {save_name}')
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        
        elif mode == "Full Training (Directory)":
            if not data_dir:
                st.error("Enter a directory path")
            elif not os.path.isdir(data_dir):
                st.error(f"Directory not found: {data_dir}")
            elif not prompts:
                st.error("Enter at least one prompt")
            else:
                # Save checkpoint if provided
                checkpoint_path = None
                if checkpoint_file:
                    temp_dir = tempfile.gettempdir()
                    checkpoint_path = os.path.join(temp_dir, checkpoint_file.name)
                    with open(checkpoint_path, "wb") as f:
                        f.write(checkpoint_file.getbuffer())
                
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_short = "_".join([p[:10] for p in prompts[:2]])
                save_name = f"clipstyler_full_{prompt_short}_{ts}.ckpt"
                
                with st.spinner("Full training... this may take a while"):
                    try:
                        model = train(
                            paths=data_dir,
                            prompts=prompts,
                            patch_size=patch_size,
                            scale_data=scale_data,
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            n_patches=n_patches,
                            img_size=img_size,
                            lambda_clip=lambda_clip,
                            lambda_patch=lambda_patch,
                            lambda_content=lambda_content,
                            lambda_tv=lambda_tv,
                            lr=lr,
                            save_name=save_name,
                            accumulate_grad_batches=accumulate,
                            compile_model=compile_model,
                            use_logger=True,
                            loader_process=loader_workers,
                            n_devices=n_devices,
                            checkpoint_path=checkpoint_path,
                            restart_epochs=10,
                        )
                        st.session_state.done = True
                        st.success(f"✅ Done! Checkpoint: {save_name}")
                        print(f'✅ Training complete! Checkpoint saved as {save_name}')
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        
        elif mode == "Predict (Inference)":
            if not checkpoint_file:
                st.error("Load a checkpoint first")
            elif not uploaded_files:
                st.error("Upload image(s)")
            else:
                image_paths = []
                temp_dir = tempfile.gettempdir()
                for file in uploaded_files:
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    image_paths.append(temp_path)
                
                # Save checkpoint
                checkpoint_path = os.path.join(temp_dir, checkpoint_file.name)
                with open(checkpoint_path, "wb") as f:
                    f.write(checkpoint_file.getbuffer())
                
                with st.spinner("Running inference..."):
                    try:
                        from utils_t import CLIPStyler, UNet
                        import torch
                        
                        model = CLIPStyler.load_from_checkpoint(
                            checkpoint_path,
                            model=UNet(),
                            loss_fn=None,
                            lr=5e-4,
                            T_0=10
                        )
                        model.eval()
                        
                        from PIL import Image
                        results = []
                        for img_path in image_paths:
                            img = Image.open(img_path).convert("RGB")
                            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            
                            with torch.no_grad():
                                output = model(img_tensor)
                            
                            output_img = (output[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            results.append((img_path, output_img))
                        
                        st.success("✅ Prediction complete!")
                        for orig_path, output_arr in results:
                            col_in, col_out = st.columns(2)
                            with col_in:
                                st.image(orig_path, caption="Input", use_container_width=True)
                            with col_out:
                                st.image(output_arr, caption="Stylized", use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")

with col_status:
    if st.session_state.done:
        st.info("✅ Complete! Check `./checkpoints/` for model")
    else:
        st.info("Ready to go!")

st.divider()
st.markdown("""
**Logs**: `./logs/` | **Checkpoints**: `./checkpoints/` | **Help**: See sidebar
""")
