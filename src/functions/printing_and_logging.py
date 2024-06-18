import logging

from colorama import Fore
from colorama import Style

import definitions

def print_and_log(text, colour):
    """
    Print the given text with the specified color and log it accordingly.

    Steps:
    1. Convert the input text to a string.
    2. If a color is specified:
        a. Print the text with the specified color using the `Fore` module from `colorama`.
        b. If the color is 'RED':
            i. Log the text as an error using `logging.error()`.
            ii. Try to post the error to an API using `api_post_string()`.
            iii. If an exception occurs during the API post, log the error using `logging.error()`.
        c. Otherwise, log the text as information using `logging.info()`.
    3. If no color is specified:
        a. Print the text without any color.
        b. Log the text as information using `logging.info()`.

    Parameters:
        text (str): The text to be printed and logged.

        colour (str): The color to be used for printing the text. Can be 'RED' or any other color supported by the `Fore` module from `colorama`.

    Returns:
        None
    """
    text = str(text)
    if colour:
        print(getattr(Fore, colour) + text + Style.RESET_ALL)
        if colour == 'RED':
            logging.error(text)
        else:
            logging.info(text)
    else:
        print(text)
        logging.info(text)

def print_load():
    """

    Prints a load message.

    """
    print_and_log("""\

                     _             
                    (_)            
  ___  ___ ___  _ __ _ _ __   __ _ 
 / __|/ __/ _ \| '__| | '_ \ / _` |
 \__ \ (_| (_) | |  | | | | | (_| |
 |___/\___\___/|_|  |_|_| |_|\__, |
                              __/ |
                             |___/ 
   _____ _____ ________  __  ____  
  / ____|_   _|___  /  \/  |/ __ \ 
 | |  __  | |    / /| \  / | |  | |
 | | |_ | | |   / / | |\/| | |  | |
 | |__| |_| |_ / /__| |  | | |__| |
  \_____|_____/_____|_|  |_|\____/ 
  _                     _          
 | |                   | |         
 | |     ___   __ _  __| |         
 | |    / _ \ / _` |/ _` |         
 | |___| (_) | (_| | (_| |_ _ _    
 |______\___/ \__,_|\__,_(_|_|_)   
                                   
                                   

    """, 'GREEN')


def print_train():
    """

    Prints a train message.

    """
    print_and_log("""\

                     _             
                    (_)            
  ___  ___ ___  _ __ _ _ __   __ _ 
 / __|/ __/ _ \| '__| | '_ \ / _` |
 \__ \ (_| (_) | |  | | | | | (_| |
 |___/\___\___/|_|  |_|_| |_|\__, |
                              __/ |
                             |___/ 
   _____ _____ ________  __  ____  
  / ____|_   _|___  /  \/  |/ __ \ 
 | |  __  | |    / /| \  / | |  | |
 | | |_ | | |   / / | |\/| | |  | |
 | |__| |_| |_ / /__| |  | | |__| |
  \_____|_____/_____|_|  |_|\____/ 
  _______        _                 
 |__   __|      (_)                
    | |_ __ __ _ _ _ __            
    | | '__/ _` | | '_ \           
    | | | | (_| | | | | |_ _ _     
    |_|_|  \__,_|_|_| |_(_|_|_)    
                                   
                                   

    """, 'GREEN')


def print_eval():
    """

    Prints an eval message.

    """
    print_and_log("""\

                     _                                  
                    (_)                                 
  ___  ___ ___  _ __ _ _ __   __ _                      
 / __|/ __/ _ \| '__| | '_ \ / _` |                     
 \__ \ (_| (_) | |  | | | | | (_| |                     
 |___/\___\___/|_|  |_|_| |_|\__, |                     
                              __/ |                     
   _____ _____ ________  __  |___/                      
  / ____|_   _|___  /  \/  |/ __ \                      
 | |  __  | |    / /| \  / | |  | |                     
 | | |_ | | |   / / | |\/| | |  | |                     
 | |__| |_| |_ / /__| |  | | |__| |                     
  \_____|_____/_____|_|  |_|\____/  _                   
 |  ____|        | |           | | (_)                  
 | |____   ____ _| |_   _  __ _| |_ _  ___  _ __        
 |  __\ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \       
 | |___\ V / (_| | | |_| | (_| | |_| | (_) | | | |_ _ _ 
 |______\_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_(_|_|_)
                                                        
                                                        
    """, 'GREEN')


def print_end():
    """

    Prints an end message.

    """
    print_and_log("""\

     _      ______ _   _ _____       _    
  /\| |/\  |  ____| \ | |  __ \   /\| |/\ 
  \ ` ' /  | |__  |  \| | |  | |  \ ` ' / 
 |_     _| |  __| | . ` | |  | | |_     _|
  / , . \  | |____| |\  | |__| |  / , . \ 
  \/|_|\/  |______|_| \_|_____/   \/|_|\/ 
                                          
                                          

    """, 'GREEN')
