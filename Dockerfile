FROM nginx:alpine
COPY site/ /usr/share/nginx/html/
COPY docker/40-chilitrack-website-id.sh /docker-entrypoint.d/40-chilitrack-website-id.sh
RUN chmod +x /docker-entrypoint.d/40-chilitrack-website-id.sh
EXPOSE 80
