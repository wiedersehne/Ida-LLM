import requests

# 1. Get auth token

auth_url = 'https://auth.test.cluster.edrilling/auth/realms/edrilling/protocol/openid-connect/token'
payload = f'grant_type=password&client_id=wellplanner&username=ty@edrilling.no&password=9402'
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
token = ''

try:
    response = requests.post(
        url=auth_url,
        data=payload,
        headers=headers,
        verify=False
    )
    response.raise_for_status()
    json_response = response.json()
    token = json_response.get('access_token')
    # print('Access Token:', token)
except requests.exceptions.RequestException as e:
    print('An error occurred while getting the auth token:', e)


# 2. Convert a simulation configuration
zip_name = 'wellControlSimulation.zip'
converter_url = 'https://wellplanner.test.cluster.edrilling/v1.7/converter/convertfromedh'
well_config_json = ''

with open(zip_name, 'rb') as file_data:
    try:
        headers = {
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/zip'
        }
        response = requests.post(
            url=converter_url,
            data=file_data,
            headers=headers,
            verify=False
        )

    except Exception as ex:
        print('ConvertFromEdh - Exception occured: ' + str(ex))
        print('An error occurred while converting the well config file:', ex)
    if response.status_code != 201 & response.status_code != 200:
        print(f'ConvertFromEdh - Non-201/Non-200 occurred: {response.status_code}: {response.text}')
        print('An error occurred while converting the well config file:', response.text)
   
    well_config_json = response.content

# print(well_config_json)


# 3. Start the simulation
simulation_url = 'https://wellplanner.test.cluster.edrilling/v1.7/wce'

headers = {
    'Authorization': 'Bearer ' + token,
    'Content-Type': 'application/json'
}
response = requests.post(
    url=simulation_url,
    data=well_config_json,
    headers=headers,
    verify=False
)

print(response)
print('Simulation successfully started')