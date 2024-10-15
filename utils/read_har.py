import json
from pprint import pprint
import requests
from urllib.parse import quote, unquote
import os
from pathlib import Path


def get_selenium_path():
    current_folder_path = Path().resolve()
    current_folder_name = current_folder_path.name
    while current_folder_name != "selenium":
        current_folder_path = current_folder_path.parent
        current_folder_name = current_folder_path.name
    return current_folder_path


def construct_folder(parent_folder):
    selenium_folder_path = str(get_selenium_path())
    request_folder_path = os.path.abspath(os.path.join(selenium_folder_path, f"json/{parent_folder}/request"))
    response_folder_path = os.path.abspath(os.path.join(selenium_folder_path, f"json/{parent_folder}/response"))

    if not os.path.exists(request_folder_path) \
            and not os.path.exists(response_folder_path):
        os.makedirs(request_folder_path, exist_ok=True)
        os.makedirs(response_folder_path, exist_ok=True)

    return request_folder_path, response_folder_path


def write_dict_to_json_file(dict_data, json_file):
    with open(json_file, "w", encoding="utf8") as file:
        json.dump(dict_data, file, ensure_ascii=False, indent=4)
        

def write_data_to_text(text, file_path):
    with open(file_path, "w", encoding="utf8") as file:
        file.write(text)


def read_data_network(filename):
    # Load the HAR file
    with open(filename, 'r', encoding="utf8") as file:
        har_data = json.load(file)

    # Access the entries in the HAR file
    entries = har_data['log']['entries']

    return entries


def find_list_request_have_text(list_text, entries):
    # Extract and print URLs and response status codes
    entries_find = []
    for entry in entries:
        request = entry['request']
        request_str = str(request)
        
        response = entry['response']
        response_str = str(response)

        for text in list_text:
            if text in response_str or text in request_str:
                entries_find.append(entry)
    return entries_find


def construct_name_value(data):
    data_converted = {}
    for item in data:
        # if item["name"][0] == ":":
        #     continue
        data_converted[item["name"]] = item["value"]
    return data_converted


def inspect_entry(entry, index=0, output="file", request_folder_path=None, response_folder_path=None):
    if output == "console":
    
        pprint(entry["response"], indent=3)
        print("==================================================================================")
        pprint(entry["request"], indent=3)
    
    elif output == "file":
        write_dict_to_json_file(entry["request"], f"{request_folder_path}/request_{index}.json")
        write_dict_to_json_file(entry["response"], f"{response_folder_path}/response_{index}.json")


def inspect_json_data(data, index=0, output="file", response_folder_path=None):
    if type(data) is str:
        data = json.loads(data)
    
    if output == "console":
        pprint(data["response"], indent=3)
    
    elif output == "file":
        write_dict_to_json_file(data, f"{response_folder_path}/response_{index}.json")


if __name__ == "__main__":
    PARENT_FOLDER = "vietstock"
    REQUEST_FOLDER_PATH, RESPONSE_FOLDER_PATH = construct_folder(PARENT_FOLDER)

    # Access the entries in the HAR file
    entries = read_data_network(
        f"./network_har/{PARENT_FOLDER}/vietstock.vn.har")

    entries_find = find_list_request_have_text(
        "Sửa quy định về hoạt động của nhà đầu tư nước ngoài trên thị trường chứng khoán Việt Nam", entries)
    write_dict_to_json_file(
        entries_find, f"{REQUEST_FOLDER_PATH}/entries_find.json")

    len(entries_find)

    list_product = entries_find[0]
    request_of_list_product_api = list_product["request"]
    product = json.loads(list_product["response"]["content"]["text"])
    write_dict_to_json_file(product, "product_iphone.json")

    write_dict_to_json_file(request_of_list_product_api, "request_data.json")

    headers = construct_name_value(request_of_list_product_api["headers"])

    headers_keys = list(headers.keys())[:]
    for key in headers_keys:
        if key[0] == ":":
            del headers[key]

    cockies = construct_name_value(request_of_list_product_api["cookies"])

    response = requests.get(
        url=request_of_list_product_api["url"],
        headers=headers,
        cookies=cockies
    )

    write_dict_to_json_file(response.json(), "product_iphone_manual_get.json")

    url = request_of_list_product_api["url"]

    # Encode the URL
    encoded_url = quote(url, safe='')

    # Print the encoded URL
    print(encoded_url)

    # Encode the URL
    encoded_url = quote("máy giặt", safe=' ')

    # Print the encoded URL
    print(encoded_url)

    decoded_url = unquote(url)

    # Print the decoded URL
    print(decoded_url)
