import os
import torch
import gc
from diffusers import Cosmos3OmniPipeline

# 1. Load the Text-to-Image specific NF4 pipeline
print("Loading Cosmos3-Super-Text2Image-nf4 model...")
pipe = Cosmos3OmniPipeline.from_pretrained(
    "SanDiegoDude/Cosmos3-Super-Text2Image-nf4",
    torch_dtype=torch.bfloat16,
    enable_safety_checker=False
).to("cuda")

# 2. Define the 5 Enhanced Prompts
prompts = [
    # Prompt 1: Police Escort with Traffic Cones
    "In the heart of a bustling urban landscape, a sleek black Toyota Land Cruiser glides through a meticulously arranged chicane, its robust frame a testament to power and precision. The camera, mounted on the dash, captures the scene with a steady, unblinking gaze, framing the vehicle from a slightly elevated angle that accentuates its commanding presence. The narrow avenue stretches into the distance, flanked by towering, monochromatic buildings that reflect the harsh midday sun, casting sharp, angular shadows that dance across the concrete. The atmosphere is thick with the hum of traffic and the distant murmur of the city, while the air shimmers with the heat of the day, enhancing the stark contrast between the black asphalt and the bright, reflective surfaces of the surrounding vehicles. The Land Cruiser, a beacon of authority, is flanked by two police escort vehicles, their rooftop strobe lights pulsating with an intense red and blue glow, illuminating the scene with a dramatic chiaroscuro effect. The light bounces off the chrome accents of the Land Cruiser, creating a mesmerizing interplay of reflections that draws the eye. The vehicles are separated by a precise distance, their steady pace creating a harmonious rhythm that underscores the importance of their mission. The chicane, a serpentine path defined by bright orange reflective cones, stretches out before the Land Cruiser like a gauntlet. The cones, their surfaces glistening under the harsh light,",

    # Prompt 2: Massive Landslide
    "In the heart of a storm-tossed mountain pass, a first-person dashcam captures the world through a rain-slicked windshield, the wipers slicing through a relentless deluge that blurs the line between sky and earth. The camera, fixed in a steady gaze, frames a narrow, winding road clinging to the edge of a steep, rocky cliff, its edges crumbling into a chasm of shadows. The air is thick with mist, backlit by distant lightning that casts an eerie glow on the saturated greens of the dense forest pressing in from the left, while the right reveals a breathtaking plunge into a valley shrouded in storm clouds. The camera, positioned at a slight downward angle, accentuates the drama of the scene as it follows the rhythmic dance of the wipers, their rhythmic swish revealing a world transformed by the elements. The tarmac glistens like obsidian under the relentless assault of rain, reflecting the kaleidoscope of headlights from oncoming vehicles, their taillights a distant, blurred ribbon of red in the gloom. The depth of field is shallow, drawing the eye to the chaos unfolding ahead, where a massive landslide has torn loose from the cliffside, sending a colossal wave of earth and stone cascading onto the road. Boulders, their surfaces glistening with mud and rain, tumble and crash in slow motion, while a roiling tide",

    # Prompt 3: Damaged Highway
    "In the heart of a somber, overcast afternoon, the camera stands steadfast at eye level, capturing a gritty tableau of a highway in disarray. The asphalt, a patchwork of jagged cracks and deep, ominous potholes, stretches into a receding vanishing point, framed by the rusted guardrails that struggle to contain the chaos. The air is thick with a palpable tension, enhanced by the muted light filtering through the heavy clouds above, casting a gloomy pallor over the scene. The road surface, a mosaic of dark, weathered asphalt and stark white roadwork markings, glistens with a thin film of rain, reflecting the desolate landscape like a broken mirror. A motley assortment of vehicles navigates this treacherous path, their tires crunching over loose gravel and chunks of broken pavement. A dilapidated sedan, its blue paint faded and chipped, limps along the left, its driver leaning forward, eyes scanning the road like a wary animal. Beside it, a sleek, black SUV, its windows tinted, navigates with a cautious precision, its tires barely whispering against the debris. Further ahead, a battered pickup truck, its bed filled with tools and supplies, lurches over a particularly deep pothole, sending a spray of gravel into the air. In the foreground, a lone figure, clad in a reflective vest, crouches beside a battered",

    # Prompt 4: Drone Aerial Landslide
    "In a breathtaking high-angle cinematic drone shot, the camera hovers majestically above a steep, rugged mountain ridge, capturing the dramatic moment of a catastrophic slope failure. The sun casts a golden-hour glow, its rays piercing through the canopy, illuminating the dense forest of towering conifers and deciduous trees, their vibrant greens and warm autumn hues creating a tapestry of life against the rugged, earthy tones of the mountainside. The camera, positioned at a sharp angle, reveals the intricate textures of the terrain—raw, exposed rock faces and the soft, crumbling earth that gives way to the relentless force of gravity. The slope, a patchwork of dark, loamy soil and jagged rock outcrops, begins to shift, a slow, ominous rumble building as gravity asserts its power. The camera, with a shallow depth of field that blurs the distant peaks and draws the viewer's eye to the chaos unfolding below, frames the scene with precision. As the earth gives way, a colossal wave of debris—earth, rock, and mud—sweeps down the mountainside, a living, churning mass that swallows everything in its path. A winding highway, a slender ribbon of asphalt, slices through the landscape, its two narrow lanes framed by guardrails that seem pitifully inadequate against the impending force. The camera captures the moment the debris flow reaches the road, the earth cascading over the",

    # Prompt 5: Falling Cargo Hazard
    "In a cinematic still, the camera, a dashcam, captures a dramatic moment from a forward-facing perspective, closely trailing a rugged, three-wheeled cargo tempo as it navigates a bustling multi-lane highway. The tempo, a rustic amalgamation of metal and wood, its once vibrant blue paint now faded and chipped, is a testament to its arduous journeys. The rear bed, a patchwork of weathered wood, is lined with a motley assortment of cardboard boxes, their sizes and states of wear hinting at a myriad of destinations. The tempo's driver, clad in a worn, brown leather jacket, grips the steering wheel with weathered hands, his expression a mix of determination and weariness, framed by the cracked side mirror. The highway stretches out in a symphony of concrete and steel, a ribbon of tarmac glistening under the harsh, midday sun, its surface a tapestry of faded white lines and scattered debris. To the left, a towering concrete barrier looms, its graffiti-scarred face a stark contrast to the ordered chaos of the road. To the right, a sprawling industrial complex hums with life, its smokestacks spewing plumes into the smog-laden air, while a faded billboard, its message long since erased, sags forlornly in the background. Suddenly, a cardboard box, its flaps loosely secured, teeters on"
]

# 3. Batch Image Generation Loop
output_dir = "enhanced_generations"
os.makedirs(output_dir, exist_ok=True)

print(f"\nStarting generation of {len(prompts)} images...\n")

for idx, prompt_text in enumerate(prompts, start=1):
    output_path = os.path.join(output_dir, f"generation_{idx}.png")
    print(f"[{idx}/{len(prompts)}] Generating: {output_path}...")
    
    # Generate the image
    result = pipe(
        prompt=prompt_text,
        height=1024,
        width=1024,
        num_frames=1,
        num_inference_steps=35,
        guidance_scale=4.0
    )
    
    # Save the generated image
    image = result.video[0]
    image.save(output_path)
    print(f"Saved: {output_path}\n")
    
    # Clear CUDA cache between generations to prevent memory buildup
    torch.cuda.empty_cache()
    gc.collect()

print("All images successfully generated!")