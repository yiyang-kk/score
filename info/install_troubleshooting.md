## Installation troubleshooting

**_`conda` - add to path_, for easy installation of prerequisites.**

There are sometimes problems with `conda`.
- The easy way to install prerequisities is to start `Anaconda Prompt` (Using START menu), and type them in this prompt. 
    - I personally discourage this - **it blocks systematic solutions with installation (i.e. one-click install scripts)**
- **To be consistent, it is important to have `conda` in `PATH`.**
    - **This can be either done in the Anaconda installation (the easy way - you can always reinstall Anaconda)**
        - during the install, you have to tick the option `Add Anaconda to PATH`.
    - or
    with following command - to be put into command line:
    ```
    SETX PATH "%PATH%;C:\PATH_WHERE_ANACONDA_IS_INSTALLED\Continuum\anaconda3\;C:\PATH_WHERE_ANACONDA_IS_INSTALLED\Continuum\anaconda3\Scripts;C:\PATH_WHERE_ANACONDA_IS_INSTALLED\Continuum\anaconda3\Library\bin;C:\PATH_WHERE_ANACONDA_IS_INSTALLED\Continuum\anaconda3\Library\mingw-w64\bin;C:\PATH_WHERE_ANACONDA_IS_INSTALLED\Continuum\anaconda3\Library\usr\bin;
    ```
    - In my case, `PATH_WHERE_ANACONDA_IS_INSTALLED` was `C:\Users\jan.hynek\AppData\Local\`

    - Then every command line (`cmd`, PowerShell or `git bash`) registers `conda` command
- **Benefits:**
    - When the `conda` command is registered, we can use `requirements.txt`, and log all needed requirements for future smooth use - including needed versions (such as `qgrid>=1.0.3`). This extends also for other repositories.
    - Afterwards, we can create single script for installing prerequisites as well as making interactive grouping to work. See install script.
- **Endline: Basic installation fom further on could be performed using running single script, thus saving time.**
