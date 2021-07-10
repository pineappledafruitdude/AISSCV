FROM alpine:3.14
RUN apk add --no-cache git 
WORKDIR /

COPY start.sh /scripts/start.sh
RUN ["chmod", "+x", "/scripts/start.sh"]
ENTRYPOINT ["sh","/scripts/start.sh", "-darknet=/darknet", "-path=/"]