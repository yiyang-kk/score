# Front-end 

1) Use in Python

    - The front end is called in jupyter notebook through function

    ```
    nfe.notebook_app.notebook_front_end
    ```

    - This function has following parameters:

        - data {pd.DataFrame} -- data for which metadata should be calculated and displayed.
                                 These data are just quickly processed - they are not displayed
                                 in the front-end wholly. The processing however can take a while
        - host {str}, default '127.0.0.1' -- either 'auto' or ip address
                                 if auto, ip address of the current server is obtained.
                                 Otherwise, this argument is fed to `dash.app` arguments.
        - port {int}, default 8051 -- port on which the app should be running
        - kwargs - key-word arguments to be passed to `dash.app`
    
    - This function starts the interactive front-end.
    - It is actually just a wrapper for several subfunctions
        - metadata creation
        - callback registration (from metadata)
        - calling `dash.app`
    - When you would like to continue after the front-end, interrupt the kernel.
                        
2) Description of front-end

    - Simple  
        TBA
    - Ratio  
        TBA
    - TimeSince  
        TBA
    - Config generation  
        TBA