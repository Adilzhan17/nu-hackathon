package httpserver

import (
	"net/http"
	"time"
)

func New(addr string, handler http.Handler, uploadTimeout time.Duration) *http.Server {
	return &http.Server{Addr: addr, Handler: handler, ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: uploadTimeout, WriteTimeout: uploadTimeout + 10*time.Second,
		IdleTimeout: 60 * time.Second, MaxHeaderBytes: 32 << 10}
}
