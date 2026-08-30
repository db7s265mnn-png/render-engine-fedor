#include "render/bdpt_scratch.h"

namespace sol {

BdptScratch& bdptThreadScratch() {
    static thread_local BdptScratch scratch;
    return scratch;
}

}  // namespace sol
