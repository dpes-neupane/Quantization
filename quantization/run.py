import torch
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import os
import random
from torcheval.metrics import PeakSignalNoiseRatio
from .evaluation import SSIM


device = "cuda" if torch.cuda.is_available() else "cpu"
bsd100_prompts = {
           "3096.png" : "Green airplane flying",
           "8023.png" : "a bird with black stripes",
           "12084.png" : "red and white patterns",
           "14037.png" : "scenic view of mountains and rocks, and man on the top of the rock hill",
           "16077.png" : "camel drinking water",
           "19021.png" : "cactus flower",
           "21077.png" : "white and red race cars with drivers",
           "24077.png" : "two white statues on a staircase",
           "33039.png" : "artistic stone wall",
           "37073.png" : "a person walking away from fighter jet",
           "38082.png" : "A bull elk stands in a field of bluish-gray sagebrush and tall grass.",
           "38092.png" : "A herd of bison on a grassy hill.",
           "41033.png" : "Two bison calves lying on the ground",
           "41069.png" : "A marmot sits among large rocks.",
           "42012.png" : "A cougar standing among aspen trees in the snow.",
           "42049.png" : "A large hawk perches on a bare tree branch",
           "43074.png" : "A colorful pheasant with a long tail walks through tall, dry grass",
           "45096.png" : "A scuba diver explores deep, dark ocean with colorful sea anemones",
           "54082.png" : "Two adobe ovens stand beside a leafless tree under a blue sky",
           "55073.png" : "A moss-covered stone statue stands in front of a stone wall, with a small hut and trees behind it",
           "58060.png" : "A colorful striped sack is open on the ground, filled with yellow corn kernels. A wicker basket containing what looks like dark seeds sits next to it.",
           "62096.png" : "A person is windsurfing on a sunny body of water",
           "65033.png" : "A group of women in light-colored dresses holding parasols walk through a grassy area beneath a large, weeping willow tree.",
           "66053.png" : "Three light-pink piglets are standing in a small group in a muddy, grassy area. A larger, dark-colored pig is partially visible in the background, near a wooden fence",
           "69015.png" : "A gray koala is seen clinging to the trunk of a eucalyptus tree.",
           "69020.png" : "A kangaroo is lying on its side on a grassy ground. Its long hind legs are stretched out.",
           "69040.png" : "A wallaby is partially hidden in dry, tall grass. It has light-brown fur and a white patch on its belly, and it is looking directly at the camera",
           "76053.png" : "A golden pagoda in the landscape",
           "78004.png" : "A white rowboat is docked in front of a modern building.",
           "85048.png" : "A man wearing a straw hat is kneeling on a dirt surface. He is smiling and looking at the camera while working on laying down square paving stones.",
           "86000.png" : "Red tulips in front of a modern building.",
           "86016.png" : "A gravel garden with raked patterns and a small, green mound",
           "86068.png" : "Two fish swimming in clear, shallow water",
           "87046.png" : "A desert lizard with striped patterns is resting on a rock among small, smooth stones",
           "89072.png" : "A man in a park ranger uniform and hard hat points at a map",
           "97033.png" : "A snow-covered field with a red barn in the background",
           "101085.png": "Three tall, carved stone statues stand in a row.",
           "101087.png": "A person in traditional clothing holds a spear by a lake",
           "102061.png": "A chateau with ornate spires is reflected in the surrounding moat",
           "103070.png": "Two blue-footed boobies on a rocky ground",
           "105025.png": "A male lion with a full mane is sleeping with its head resting on a large rock",
           "106024.png": "A penguin stands on a rock with its head and beak raised to the sky, appearing to call out. The background is a mix of blurred ice and water",
           "108005.png": "A tiger is standing in a forest, looking at the camera",
           "108082.png": "A tiger is resting on the ground in a forest",
           "108070.png": "A tiger is wading in a small stream in a forest",
           "109053.png": "A coyote with its tongue out is walking through a forest with fallen leaves",
           "119082.png": "A large mural of a man in a tuxedo holding a violin is painted on the side of a tall building",
           "123074.png": "A golden-mantled ground squirrel stands on a rock",
           "126007.png": "A Baroque-style church with twin steeples sits in the foreground, with a large, jagged mountain range in the background under a blue sky",
           "130026.png": "A crocodile is resting on a muddy bank",
           "134035.png": "A leopard sits in a tree, looking to the side",
           "143090.png": "A tufa formation stands in a still lake at dusk, with its reflection visible in the water",
           "145086.png": "A man sits in a bleacher adorned with a red and white striped banner, watching a procession",
           "147091.png": "A tree with a windswept look sits on a hillside against a cloudy blue sky",
           "148026.png": "A reflection of a building is visible in a pond below a small wooden bridge",
           "148089.png": "The Washington Square Arch stands in a park area, surrounded by buildings. People are visible at the base of the arch",
           "156065.png": "A female scuba diver is exploring a deep-sea cavern, surrounded by large, fan-shaped coral.",
           "157055.png": "A man and woman are sitting on a porch, each holding a glass of white wine, and talking to each other",
           "159008.png": "Three fox kits are huddled together. The one on the left is looking toward the camera with its tongue slightly out",
           "160068.png": "A clouded leopard is lounging on a thick tree branch",
           "163085.png": "Three young, fuzzy bird chicks are huddled together in a nest made of straw",
           "167062.png": "A wolf is walking on a snow-covered slope near a dense forest",
           "167083.png": "A clear alpine lake surrounded by mountains and a rock face",
           "170057.png": "Two people in military fatigues and blue helmets are walking in a field",
           "175032.png": "A slender, legless lizard is on the ground near a stick and some pine needles.",
           "175043.png": "A small, bright-green snake is eating a brown grasshopper on a gravelly surface. The snake has its mouth open wide to consume the insect",
           "182053.png": "A train on a viaduct in a snowy landscape.",
           "189080.png": "A man with a red hat and a slight smile",
           "196073.png": "A snake is on the sand, facing a small insect",
           "197017.png": "Three brown horses with thick manes are standing in a field",
           "208001.png": "A light-colored mushroom with a wide cap and a tall stem is standing among green plants and brown leaves on the forest floor",
           "210088.png": "A small, light orange fish with a white stripe near its eye is partially hidden among the waving tentacles of an anemone",
           "216081.png": "Two people in cowboy attire are standing in front of a covered wagon",
           "219090.png": "A view of a Norwegian harbor town with a small boat and people on a pier in the foreground",
           "220075.png": "A person on horseback, wearing a cowboy hat and a vest, is overlooking a herd of cattle in a pen",
           "223061.png": "A low-angle view of the Louvre Pyramid, with a cloudy sky in the background",
           "227092.png": "A large, ancient, light-colored ceramic vase stands on a polished floor. It has two handles and is decorated with geometric and spiral patterns.",
           "229036.png": "Two men in traditional outfits with a drum and cymbals.",
           "236037.png": "Two river otters are on a muddy bank with green plants",
           "241004.png": "A field with several large, moss-covered boulders is in the foreground, with rolling, tree-covered hills in the background",
           "241048.png": "A river flows through a valley with hills",
           "253027.png": "A family of zebras is standing in a green field with some scattered trees in the background",
           "253055.png": "A group of giraffes walks across a grassy plain",
           "260058.png": "The Giza pyramids are visible in the distance, with a vast expanse of sand in the foreground",
           "271035.png": "A man with a yoke carries two containers",
           "285079.png": "A firefighter in full gear with a blue helmet is sifting through debris at the site of a fire",
           "291000.png": "A brown horse with a black mane is standing in a green field with white blossoms on the trees in the background",
           "295087.png": "A desert landscape features a tree in the foreground and a natural rock arch in the background",
           "296007.png": "A single buffalo is standing in a large, dry field. In the background, there are a few trees and a range of hazy mountains",
           "296059.png": "Two adult elephants are standing in a shallow body of water in a grassy field.",
           "299086.png": "A person rides a camel in the desert near a pyramid",
           "300091.png": "A surfer is riding inside the tube of a large wave, with land visible in the background",
           "302008.png": "A man in a striped shirt and black jacket.",
           "304034.png": "A large black leopard is lying on a bank in a wooded area, with its mouth slightly open",
           "304074.png": "A ram with very large, curved horns is standing on a rocky outcrop",
           "306005.png": "Two Moorish idol fish are swimming near a coral reef.",
           "351093.png": "A steam locomotive is crossing a stone arch bridge over a river. The bridge and river are surrounded by dense green trees",
           "361010.png": "Two polo players on horseback during a match",
           "376043.png": "A person in a military uniform is sitting on the grass",
           "385039.png": "A man is kneeling to milk a white cow in front of a white building with a red roof. Another man stands nearby, holding the cow's lead"
           }


def load_img(filename, resize=True):
  low_res_img = Image.open(filename).convert("RGB")
  if resize:
    low_res_img = low_res_img.resize((128, 128))
  return low_res_img


def diffusion_loop(unet, pipeline, low_res_img, prompt, batch_size=1):
    #encode prompts
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, device='cpu')
    #prepare image
    low_res_img = pipeline.image_processor.preprocess(low_res_img)
    #set timesteps
    pipeline.scheduler.set_timesteps(75)
    timesteps = pipeline.scheduler.timesteps
    #add noise to image
    noise_level = torch.tensor([20], dtype=torch.long)
    noise = torch.randn(low_res_img.shape)
    image = pipeline.low_res_scheduler.add_noise(low_res_img, noise, noise_level)
    # noise_level = torch.cat([noise_level] * image.shape[0])
    #prepare latents
    num_channels_latents = pipeline.vae.config.latent_channels
    latents = torch.randn((batch_size, num_channels_latents, low_res_img.shape[2], low_res_img.shape[3]))
    latents = latents * pipeline.scheduler.init_noise_sigma

    context = torch.cat([negative_prompt_embeds, prompt_embeds])
    # context = prompt_embeds

    image = torch.cat([image] * 2)
    noise_level = torch.cat([noise_level]*image.shape[0])
    # print(noise_level, image.shape[0], [noise_level] * image.shape[0])

    latents = latents.to(device)
    context = context.to(device)
    image = image.to(device)
    noise_level = noise_level.to(device)
    guidance_scale = 9.0
    unet.to(device)
        
    for i, t in enumerate(tqdm(timesteps)):
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            # latent_model_input = latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, image], dim=1)
            noise_pred = unet(latent_model_input,
                                t,
                                encoder_hidden_states=context,
                                class_labels=noise_level,
                                return_dict=False)[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents


def saveImg(pipeline, latents, save_folder):
    pipeline.vae.to(device)
    upscaled_image = pipeline.vae.decode(latents/ pipeline.vae.config.scaling_factor, return_dict=False)[0]
    upscaled_image = pipeline.image_processor.postprocess(upscaled_image.cpu().detach())

    # upscaled_image = upscaled_image.permute(0, 2, 3, 1).float().detach().to('cpu').numpy()
    # upscaled_image = pipeline.image_processor.numpy_to_pil(upscaled_image)[0]
    # print(type(upscaled_image))
    print(f"The image is saved in {save_folder}")
    upscaled_image[0].save(save_folder)
    
    
    
# imgs = os.listdir("./BSD100_LR_x2")
# random.seed(43)
# random.shuffle(imgs)
# calib_set = imgs[:70]
# test_set = imgs[70:]

def run(unet, 
        pipeline, 
        img_set, 
        prompt_dict=bsd100_prompts, 
        batch_size=1, test_mode=False, 
        target_set_folder=None, 
        save_folder=None):
  if test_mode:
    psnr = PeakSignalNoiseRatio()
    ssim = SSIM()

  for i in range(0, len(img_set), batch_size):
    im1 = img_set[i]

    prompt = prompt_dict[im1]
    path1 = os.path.join('./BSD100_LR_x2', im1)
    img_ = load_img(path1)

    # img_ = [img_1, img_2]
    # prompt = [prompt1, prompt2]
    if test_mode:
      latents = diffusion_loop(unet, pipeline, img_, prompt)
      p1 = os.path.join(target_set_folder, im1)
      # p2 = os.path.join(target_set_folder, im2)
      target_img = load_img(p1, resize=False)
      # print(type(target_img))
      # target_img2 = load_img(p2)
      # latents.to('cpu')
      pipeline.vae.to(device)
      upscaled_image = pipeline.vae.decode(latents/ pipeline.vae.config.scaling_factor, return_dict=False)[0]
      upscaled_image = pipeline.image_processor.postprocess(upscaled_image.cpu().detach())[0]
      if save_folder is not None:
        upscaled_image.save(save_folder + im1)
      # print(type(upscaled_image))
      upscaled_image = upscaled_image.resize((target_img.size[0], target_img.size[1]))
    #   print(upscaled_image.size, target_img.size)
      upscaled_image = pipeline.image_processor.preprocess(upscaled_image)
      target_img = pipeline.image_processor.preprocess(target_img)
    #   print(upscaled_image.shape, target_img.shape)
      psnr.update(upscaled_image, target_img)
      ssim.update(upscaled_image, target_img)
    else: diffusion_loop(unet, pipeline, img_, prompt, batch_size)
    print(f"Total {i+1} images processed.")
  if test_mode:
    psnr_val = psnr.compute()
    ssim_val = ssim.compute()
    print(f"The psnr is: {psnr_val} and ssim is: {ssim_val}")