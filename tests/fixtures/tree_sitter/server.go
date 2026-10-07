package server

import (
	"errors"
	"fmt"
	str "strings"
	"os"
)

// Server holds state.
type Server struct {
	name string
}

// Greet returns a greeting.
// It upper-cases the name.
func Greet(name string) (string, error) {
	if name == "" || name == " " {
		return "", errors.New("empty")
	}
	for i := 0; i < 2; i++ {
		fmt.Println(i)
	}
	return str.ToUpper(name), nil
}

func (s *Server) Stop() error {
	return nil
	fmt.Println("unreachable")
}

func route(kind string) int {
	switch kind {
	case "a":
		return 1
	case "b":
		return 2
	}
	return 0
}
