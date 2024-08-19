import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog


input_folder = ""
checkbox_recognition = False
year_recognition = False
options = []


ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

app = ctk.CTk()  # create CTk window like you do with the Tk window
app.geometry("700x500")
app.title("Preprocess for OCRmyPDF")


# Function to open a file dialog and display the selected file path in the input field
def select_file():
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("All Files", "*.*")]
    )
    if file_path:
        file_input_var.set(file_path)

# Function to open a folder dialog and display the selected folder path in the input field
def select_folder():
    folder_path = filedialog.askdirectory(
        title="Select a Folder"
    )
    if folder_path:
        folder_input_var.set(folder_path)
        input_folder = folder_path
        print(input_folder)
        
def onClickCheckboxOpt():
    if check_opt_var.get() == 'on':
        options.append('checkbox')
    elif 'checkbox' in options:
        options.remove('checkbox')
    print("options:", options)
    
def onClickYearOpt():
    if year_opt_var.get() == 'on':
        options.append('year')
    elif 'year' in options:
        options.remove('year')
    print("options:", options)
###############################################################
# Create a folder_dialog_frame using grid layout manager
###############################################################

folder_dialog_frame = ctk.CTkFrame(app)
folder_dialog_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
folder_dialog_frame.grid_columnconfigure(0, weight=1)


title = ctk.CTkLabel(folder_dialog_frame, text="Preprocess for OCRmyPDF", font=("Helvetica", 24))
title.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=4)

# Create and place buttons, input fields, and labels
file_input_var = ctk.StringVar()
file_input = ctk.CTkEntry(folder_dialog_frame, textvariable=file_input_var, placeholder_text="Selected File")
file_input.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

select_file_button = ctk.CTkButton(folder_dialog_frame, text="Select File", command=select_file)
select_file_button.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

folder_input_var = ctk.StringVar()
folder_input = ctk.CTkEntry(folder_dialog_frame, textvariable=folder_input_var, placeholder_text="Selected Folder")
folder_input.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

select_folder_button = ctk.CTkButton(folder_dialog_frame, text="Select Folder", command=select_folder)
select_folder_button.grid(row=2, column=3, padx=5, pady=5, sticky="ew")

# Configure columns to expand properly
folder_dialog_frame.columnconfigure(0, weight=1)
folder_dialog_frame.columnconfigure(1, weight=1)
folder_dialog_frame.columnconfigure(2, weight=1)
folder_dialog_frame.columnconfigure(3, weight=1)



###############################################################
# Create an option_frame using grid layout manager
###############################################################

option_frame = ctk.CTkFrame(app)
option_frame.grid(row=1, column=0, padx=10, pady=(5, 5), sticky="nsew")
option_frame.grid_columnconfigure(0, weight=1)

option_title = ctk.CTkLabel(option_frame, text="Select the options for preprocessing of OCRmyPDF.", font=("Helvetica", 16), anchor="w")
option_title.grid(row=0, column=0, padx=5, pady=5, sticky="ew", columnspan=4)



check_opt_var = ctk.StringVar(value="off")
checkbox_opt = ctk.CTkCheckBox(option_frame, text="Recognition for Checkbox", command=onClickCheckboxOpt, variable=check_opt_var, onvalue="on", offvalue="off")
checkbox_opt.grid(row=1, column=0, padx=5, pady=5, sticky="ew", columnspan=4)


check_opt_var = ctk.StringVar(value="off")
checkbox_opt = ctk.CTkCheckBox(option_frame, text="Recognition for Checkbox", command=onClickCheckboxOpt, variable=check_opt_var, onvalue="on", offvalue="off")
checkbox_opt.grid(row=1, column=0, padx=5, pady=5, sticky="ew", columnspan=4)

year_opt_var = ctk.StringVar(value="off")
year_opt = ctk.CTkCheckBox(option_frame, text="Recognition for Year", command=onClickYearOpt, variable=year_opt_var, onvalue="on", offvalue="off")
year_opt.grid(row=2, column=0, padx=5, pady=5, sticky="ew", columnspan=4)
# Configure columns to expand properly
option_frame.columnconfigure(0, weight=1)
option_frame.columnconfigure(1, weight=1)
option_frame.columnconfigure(2, weight=1)
option_frame.columnconfigure(3, weight=1)

###############################################################
# Create an operation_frame using grid layout manager
###############################################################

operation_frame = ctk.CTkFrame(app)
operation_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="nsew")
operation_frame.grid_columnconfigure(0, weight=1)

# operation_title = ctk.CTkLabel(operation_frame, text="Select the options for preprocess of OCRmyPDF.", font=("Helvetica", 16), anchor="w")
# operation_title.grid(row=0, column=0, padx=5, pady=5, sticky="ew", columnspan=4)

def onClickPreprocess():
    print("Process button pressed")

btn_preprocess = ctk.CTkButton(operation_frame, text="Process", command=onClickPreprocess)
btn_preprocess.grid(row=1, column=3, padx=5, pady=5, sticky="ew")











app.grid_rowconfigure(1, weight=1)
app.grid_columnconfigure(0, weight=1)
# Run the application
app.mainloop()
