FROM scilus/scilus:1.6.0

# install the code
RUN mkdir /fwdti
COPY ./fit_fw* /fwdti/

# set run command to call the script
ENTRYPOINT ["/fwdti/fit_fw_scilpy.sh"]
