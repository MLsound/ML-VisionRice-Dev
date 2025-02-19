# ../images/raw/Eryx/101MEDIA/2 utiles/

import os
import exifread
import geopy.distance
import pandas as pd

def sanitize_folder_input(image_directory):
    """Validates and sanitizes folder input."""

    if not image_directory:
        return None, "Error: Folder path cannot be empty."

    image_directory = os.path.normpath(image_directory)
    image_directory = os.path.realpath(image_directory)

    if not os.path.isdir(image_directory):
        return None, f"Error: Directory '{image_directory}' not found."

    return image_directory, None  # Return sanitized path and no error

def get_geotagging(image_path):
    """Extracts latitude and longitude from EXIF data."""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

    try:
        lat_ref = tags['GPS GPSLatitudeRef']
        lat = tags['GPS GPSLatitude'].values  # Access the values attribute of the Ratio object
        lon_ref = tags['GPS GPSLongitudeRef']
        lon = tags['GPS GPSLongitude'].values  # Access the values attribute of the Ratio object

        # Convert to decimal degrees
        lat_deg = float(lat[0]) + float(lat[1])/60 + float(lat[2])
        lon_deg = float(lon[0]) + float(lon[1])/60 + float(lon[2])

        if lat_ref.printable == 'S':
            lat_deg = -lat_deg
        if lon_ref.printable == 'W':
            lon_deg = -lon_deg

        return (lat_deg, lon_deg)
    except KeyError:
        return None  # No geotagging information

def group_images_by_location(image_dir, threshold_distance=0.05): # 50 meters default
    """Groups images based on geospatial proximity."""
    image_locations = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg')):  # Check for JPEG files
            image_path = os.path.join(image_dir, filename)
            location = get_geotagging(image_path)
            if location:
                image_locations[filename] = location

    grouped_images = {}
    assigned_images = set()

    for image1, loc1 in image_locations.items():
        if image1 in assigned_images:
            continue

        group = [image1]
        assigned_images.add(image1)
        grouped_images[image1] = group #using the first image as the key for the group

        for image2, loc2 in image_locations.items():
            if image2 != image1 and image2 not in assigned_images:
                distance = geopy.distance.geodesic(loc1, loc2).km
                if distance <= threshold_distance:
                    group.append(image2)
                    assigned_images.add(image2)

    return grouped_images


if __name__ == "__main__":
    while True:
        image_directory = input("Enter the directory containing the images (or 'q' to quit): ")
        if image_directory.lower() == 'q':
            break

        sanitized_path, error_message = sanitize_folder_input(image_directory)

        if error_message:
            print(error_message)
            break

        image_directory = sanitized_path

        while True:

            threshold_m = int(input("Enter the threshold distance (meters): "))
            print(f"Looking for groups within {threshold_m}m around…")
            threshold = threshold_m/1000

            try:
                grouped = group_images_by_location(image_directory, threshold_distance=threshold)

                for key, images_in_group in grouped.items():
                    print(f"Group centered around: {key}")
                    for image in images_in_group:
                        print(f"  - {image}")
                    print("-" * 20)

                validation_input = input("Do you want to generate this file? [y/n]: ")

                if validation_input.lower() == 'y':
                    data = []
                    group_counter = 1
                    for key, images_in_group in grouped.items():
                        first_location = get_geotagging(os.path.join(image_directory, key))

                        if first_location:
                            lat_group, lon_group = first_location

                            for image in images_in_group:
                                image_path = os.path.join(image_directory, image)
                                location = get_geotagging(image_path)
                                if location:
                                    lat, lon = location
                                    data.append([image, lat, lon, group_counter])
                                else:
                                    print(f'Warning: No location found for {image} in group {group_counter}')
                        else:
                            print(f'Warning: No location found for first image {key} in group {group_counter}')

                        group_counter += 1

                    df = pd.DataFrame(data, columns=['filename', 'latitude', 'longitude', 'group'])

                    filename = f"geotagged_images_{threshold}km.csv"
                    df.to_csv(filename, index=False)
                    print(f"Data exported to {filename}\n")

                else:
                    print("File generation cancelled.")

                    restart = input("\nDo you want to continue processing the same direcory? [y/n]:")
                    print()
                    if restart.lower() == 'y':
                        continue
                    else:
                        break

            except FileNotFoundError:
                print(f"Error: Directory '{image_directory}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")