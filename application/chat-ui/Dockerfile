FROM node:18 AS build

WORKDIR /app

COPY application/chat-ui/package*.json ./
RUN npm ci

COPY application/chat-ui .
RUN npm run build

FROM nginx:1.25-alpine AS runtime

LABEL org.opencontainers.image.title="Medicinal Plant Chat UI" \
      org.opencontainers.image.description="Static Angular front-end for the medicinal plant classifier chatbot."

COPY application/chat-ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/chat-ui /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
