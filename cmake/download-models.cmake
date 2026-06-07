get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ggml-org/models...")
endif()

# Use an authenticated request when HF_TOKEN is set: this raises the HuggingFace
# rate limit and avoids the 429s seen when many CI jobs fetch models at once.
# Falls back to an anonymous download when the token is not available.
set(HF_AUTH_HEADER "")
if(NOT "$ENV{HF_TOKEN}" STREQUAL "")
    set(HF_AUTH_HEADER HTTPHEADER "Authorization: Bearer $ENV{HF_TOKEN}")
endif()

file(DOWNLOAD
    "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true"
    "${DEST}"
    TLS_VERIFY ON
    EXPECTED_HASH ${HASH}
    ${HF_AUTH_HEADER}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()
