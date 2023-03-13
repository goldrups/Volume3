"""Volume 3: Web Scraping.
Samuel Goldrup
Math 405
23 January 2023
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Use the requests library to get the HTML source for the website 
    http://www.example.com.
    Save the source as a file called example.html.
    If the file already exists, do not scrape the website or overwrite the file.
    """
    if not os.path.exists("./example.html"): #don't wanna overwrite that file
        jeff_source = requests.get("https://www.example.com") #named after my best friend, Jeff
        with open("example.html","w") as file:
            file.write(jeff_source.text)

    return None
    
    
# Problem 2
def prob2(code):
    """Return a list of the names of the tags in the given HTML code.
    Parameters:
        code (str): A string of html code
    Returns:
        (list): Names of all tags in the given code"""
    small_soup = BeautifulSoup(code,'html.parser')
    tags = small_soup.find_all(True) #first find their names, then
    tag_names = [tag.name for tag in tags] #get their names
    return tag_names
    

# Problem 3
def prob3(filename="example.html"):
    """Read the specified file and load it into BeautifulSoup. Return the
    text of the first <a> tag and whether or not it has an href
    attribute.
    Parameters:
        filename (str): Filename to open
    Returns:
        (str): text of first <a> tag
        (bool): whether or not the tag has an 'href' attribute
    """
    with open(filename,'r') as file:
        code = file.read() #read in the html code
    
    example_soup = BeautifulSoup(code,'html.parser') #turn it into soup
    first_a_tag = example_soup.a #get those tags
    href = False
    if hasattr(first_a_tag,"href"): #see if it has an href attribute
        href = True

    txt = first_a_tag.get_text() #this is the string
    
    return txt, href


# Problem 4
def prob4(filename="san_diego_weather.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the following tags:

    1. The tag containing the date 'Thursday, January 1, 2015'.
    2. The tags which contain the links 'Previous Day' and 'Next Day'.
    3. The tag which contains the number associated with the Actual Max
        Temperature.

    Returns:
        (list) A list of bs4.element.Tag objects (NOT text).
    """
    with open(filename,'r') as file:
        code = file.read() #read in the HTML code

    weather_soup = BeautifulSoup(code,'html.parser') #turn it in to soup

    ans_1 = weather_soup.find(string="Thursday, January 1, 2015").parent #tag with the date
    ans_2_1 = weather_soup.find(string=re.compile(r"Previous Day")).parent #previous/next day tags
    ans_2_2 = weather_soup.find(string=re.compile(r"Next Day")).parent
    ans_3 = weather_soup.find(string="59").parent #the absolute maximum temperature
   
    return [ans_1, ans_2_1, ans_2_2, ans_3] #return as a list


# Problem 5
def prob5(filename="large_banks_index.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the tags containing the links to bank data from September 30, 2003 to
    December 31, 2014, where the dates are in reverse chronological order.

    Returns:
        (list): A list of bs4.element.Tag objects (NOT text).
    """
    with open(filename,'r') as file:
        code = file.read() #read in the HTML code

    fed_soup = BeautifulSoup(code,'html.parser') #turn it into soup

    year_elements = fed_soup.find_all(string=re.compile("20(0[3-9]|1[0-4])")) #get all the 2003-2014, filter off last two
    year_tags = [y_el.parent for y_el in year_elements] #get their parents

    return year_tags[:-2]

# Problem 6
def prob6(filename="large_banks_data.html"):
    """Read the specified file and load it into BeautifulSoup. Create a single
    figure with two subplots:

    1. A sorted bar chart of the seven banks with the most domestic branches.
    2. A sorted bar chart of the seven banks with the most foreign branches.

    In the case of a tie, sort the banks alphabetically by name.
    """
    with open(filename,'r') as file:
        code = file.read() #read in HTML code


    fed_soup = BeautifulSoup(code,'html.parser') #federal reserve soup
    trs = fed_soup.find_all(name='tr')[5:1385] #get all tr tags

    #get all data for name, # of domestic banks, # of foreign banks
    names = np.array([tr.get_text().split('\n')[1] if tr.get_text() != None else None for tr in trs])
    doms = [tr.get_text().split('\n')[10] if tr.get_text() != None else None for tr in trs]
    foreigns = [tr.get_text().split('\n')[11] if tr.get_text() != None else None for tr in trs]

    #filter out the crap
    for i in range(len(doms)):
        doms[i] = doms[i].replace(",","")
        if "." in doms[i]:
            doms[i] = '0'

    for i in range(len(foreigns)):
        foreigns[i] = foreigns[i].replace(',',"")
        if "." in foreigns[i]:
            foreigns[i] = '0'

    #turn into floats yay
    doms = np.array(doms).astype(float)
    foreigns = np.array(foreigns).astype(float)

    #isolate the top seven
    ords_dom = np.argsort(doms)[::-1]
    ords_foreign = np.argsort(foreigns)[::-1]

    data_dom = doms[ords_dom][:7]
    labels_dom = names[ords_dom][:7]

    data_foreign = foreigns[ords_foreign][:7]
    labels_foreign = names[ords_foreign][:7]

    #now plot horizontal bar charts

    plt.subplot(1,2,1)
    plt.barh(labels_dom,data_dom)
    plt.yticks(fontsize=5)
    plt.title("Domestic Branches")

    plt.subplot(1,2,2)
    plt.barh(labels_foreign,data_foreign)
    plt.yticks(fontsize=5)
    plt.title("Foreign Branches")

    plt.suptitle("Federal Reserve Bank Analysis")
    plt.tight_layout()
    plt.show()


def test_1():
    print(prob1())

def test_2():
    small_example_html = """
    <html><body><p>
    Click <a id='info' href='http://www.example.com'>here</a> for more information.
    </p></body></html>
    """
    print(prob2(small_example_html))

def test_3():
    print(prob3())

def test_4():
    print(prob4())

def test_5():
    print(prob5())

def test_6():
    print(prob6())

def test_all():
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
