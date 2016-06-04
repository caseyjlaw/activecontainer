FROM caseyjlaw/rtpipe-base:latest

WORKDIR /data
RUN pip install activegit
COPY rflearn rflearn
RUN cd rflearn && python setup.py install

# jupyter boilerplate
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--notebook-dir=/data"]