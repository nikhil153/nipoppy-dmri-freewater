FROM scilus/scilus:2.0.2

# install the code
RUN mkdir -p /opt/fwdti
COPY ./fit_fw* /opt/fwdti/
RUN chmod -R 775 /opt/fwdti

# set run command to call the script
ENTRYPOINT ["/opt/fwdti/fit_fw_scilpy.sh"]
