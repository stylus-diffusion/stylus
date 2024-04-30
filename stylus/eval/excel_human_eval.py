import os
import xlsxwriter
from random import shuffle
import random

# REPLACE THIS PATH WITH THE DIRECTORY CONTAINING YOUR PROMPT DIRECTORIES
base_dir = '~/parti_realistic_output/cfg_5'

# Create a new Excel file and add a worksheet
workbook = xlsxwriter.Workbook('image_preferences.xlsx')
worksheet = workbook.add_worksheet()

# Set the column width for image columns and source flag columns
worksheet.set_column('B:C', 20)  # Adjust based on your images
worksheet.set_column('E:F', 15)  # For the source flags

# Headers
worksheet.write('A1', 'Prompt')
worksheet.write('B1', 'Image A')
worksheet.write('C1', 'Image B')
worksheet.write('D1', 'Preference')
worksheet.write('E1', 'Image A Source Flag')
worksheet.write('F1', 'Image B Source Flag')
row = 1
counter =0
# Loop through each prompt directory in the base directory
for prompt in os.listdir(base_dir):
    prompts = random.choice(os.listdir(base_dir)) 
    if counter > 150:
        break
    counter += 1
    prompt_dir = os.path.join(base_dir, prompt)

    # Directories for 'lora' and 'normal' images
    lora_dir = os.path.join(prompt_dir, 'lora')
    normal_dir = os.path.join(prompt_dir, 'normal')

    # Ensure both directories exist before proceeding
    if os.path.isdir(lora_dir) and os.path.isdir(normal_dir):
        # Filter images based on the criteria
        image_a_path = next((os.path.join(lora_dir, img) for img in os.listdir(lora_dir) if img.startswith("1")), None)
        image_b_path = next((os.path.join(normal_dir, img) for img in os.listdir(normal_dir) if img.startswith("0")), None)

        # Ensure images were found
        if image_a_path and image_b_path:
            # Shuffle the order of images
            images = [(image_a_path, '1' if 'lora' in image_a_path else '0'), 
                      (image_b_path, '1' if 'normal' in image_b_path else '0')]
            shuffle(images)

            # Assign flags based on whether the image is from the 'lora' directory
            image_a_flag = '1' if 'lora' in images[0][0] else '0'
            image_b_flag = '1' if 'lora' in images[1][0] else '0'

            # Write the prompt name d
            worksheet.write(row, 0, prompt)

            # Insert the images
            worksheet.insert_image(row, 1, images[0][0], {'x_scale': 0.5, 'y_scale': 0.5})
            worksheet.insert_image(row, 2, images[1][0], {'x_scale': 0.5, 'y_scale': 0.5})

            # Write source flags
            worksheet.write(row, 4, image_a_flag)  # Image A Source Flag
            worksheet.write(row, 5, image_b_flag)  # Image B Source Flag

            # Move to the next row
            row += 1
# Close the workbook to save it
workbook.close()