import requests
from bs4 import BeautifulSoup

url = "https://www.bing.com/search?q=what+are+different+methods+of+inventory+management"

payload = {}
headers = {
  'Cookie': 'MUID=1D542C2CF0276D6124B639D1F1CC6CA9; SRCHD=AF=NOFORM; SRCHHPGUSR=SRCHLANG=en&IG=AB99B3DB0DBF49A485324C09043526F9; SRCHUID=V=2&GUID=52B9E6A40D474A6BA787E8CC27D0180D&dmnchg=1; SRCHUSR=DOB=20250531; _EDGE_S=F=1&SID=3CEA1EEFEC8B6713001D0B12ED606645; _EDGE_V=1; _SS=SID=3CEA1EEFEC8B6713001D0B12ED606645; ak_bmsc=4C8F4128975B846E6C41E7F22792FFFE~000000000000000000000000000000~YAAQr/TfF3nguMWWAQAAHsPEJBsh1qTLZZxINCGrnWs7OmqHIrYgDgY/PeLMvexijrYAX9Zw630WMrEQDftWLqxOTtzCS/QexNriYMY6cOM+1DnCvm3j8c6kycVhuyCZoTXXUm1gL1md991ef+GFifVzNRTnhNN41woODIB1SeUKxN0sht6Q2Z5c4U6zfgXmwKseUkxdPP7aylEDRqm7i4PCIu5w9444CKUHbB4R/5I1Fnk00+4NXlVlYMH8nuojgfZ0bNuCSlO/A9du9i8lGKybd1M/0vAq8fRtN3h7oCv6FdTHnk496Izrt1nvP7Ltpg3ZFw9q8xtGWNIYdE+KFH/zVGjpsUZG2zI=; MUIDB=1D542C2CF0276D6124B639D1F1CC6CA9'
}

response = requests.request("GET", url, headers=headers, data=payload)



html = response.text
soup = BeautifulSoup(html, 'html.parser')

divs = soup.find_all('div', class_='b_tpcn')
for div in divs:
    link = div.find('a')
    if link and link.has_attr('href'):
        print(link['href'])